import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.diffusion import categorical_kl, extract, index_to_log_onehot, log_1_min_a, log_add_exp, log_categorical, log_sample_categorical, to_torch_const


class ContigousTransition(nn.Module):
    def __init__(self, sigmas, num_classes=None, scaling=1.):
        super().__init__()
        self.num_classes = num_classes
        self.scaling = scaling
        sigmas = np.asarray(sigmas, dtype=np.float64)
        if sigmas.ndim != 1:
            raise ValueError("sigmas must be a 1D array.")
        self.sigmas = to_torch_const(sigmas)

    def add_noise(self, x, time_step, batch=None):
        if batch is None:
            batch = torch.arange(time_step.shape[0]).to(time_step.device)
        if self.num_classes is not None:  # categorical values using continuous noise
            x = F.one_hot(x, self.num_classes).float()
        x = x / self.scaling
        sigma = self.sigmas.index_select(0, time_step)
        sigma = sigma.index_select(0, batch).unsqueeze(-1)
        noise = torch.randn_like(x)
        pert = x + sigma * noise
        pert = torch.where(time_step[batch, None] == 0, x, pert)
        if self.num_classes is None: # continuous values
            return pert
        else:
            return pert, x
            

    def get_prev_from_recon(self, x_t, x_recon, t, batch=None):
        if batch is None:
            batch = torch.arange(t.shape[0]).to(t.device)
        t_prev = torch.clamp(t - 1, min=0)
        sigma_t = extract(self.sigmas, t, batch)
        sigma_prev = extract(self.sigmas, t_prev, batch)
        ratio = sigma_prev / sigma_t.clamp(min=1e-12)
        time_zero = (t[batch] == 0).unsqueeze(-1)
        x_prev = ratio * x_t + (1.0 - ratio) * x_recon
        x_prev = torch.where(time_zero, x_recon, x_prev)
        return x_prev

    def sample_init(self, shape):
        if self.num_classes is None:
            return torch.randn(shape).to(self.sigmas.device)
        else:
            return torch.randn([shape, self.num_classes]).to(self.sigmas.device)


class SigmaUniformCategoricalTransition(nn.Module):
    """Categorical diffusion where sigma_t is *directly* the corruption / mask rate.

    Forward marginal:
        q(v_t | v_0) = (1 - sigma_t) * delta(v_t = v_0) + sigma_t / K

    This exactly mirrors the teacher's CategoricalPrior with prior_type="uniform":
        prob = info_level * one_hot(v_0) + (1 - info_level) * (1/K)
    using the correspondence  info_level = 1 - sigma_t.

    One-step forward transition (needed for posterior):
        q(v_t | v_{t-1}) = (1 - gamma_t) * delta(v_t = v_{t-1}) + gamma_t / K
        where gamma_t = (sigma_t - sigma_{t-1}) / (1 - sigma_{t-1})  [clipped to [0,1]]

    Posterior:
        q(v_{t-1} | v_t, p_0) propto q(v_t | v_{t-1}) * q(v_{t-1} | p_0)
        where both terms reduce to simple scalar combinations.
    """

    def __init__(self, sigmas, num_classes, init_prob=None):
        super().__init__()
        self.eps = 1e-30
        self.num_classes = num_classes
        sigmas = np.asarray(sigmas, dtype=np.float64)
        if sigmas.ndim != 1:
            raise ValueError("sigmas must be a 1D array.")
        self.sigmas = to_torch_const(sigmas)

        if init_prob is None:
            init_prob_arr = np.ones(num_classes, dtype=np.float64) / num_classes
        else:
            init_prob_arr = np.asarray(init_prob, dtype=np.float64)
            init_prob_arr = init_prob_arr / init_prob_arr.sum()
        self.register_buffer("_init_prob_buf", torch.from_numpy(init_prob_arr).float())

    # ------------------------------------------------------------------ helpers
    def _sigma(self, t, batch=None):
        """Return per-atom sigma, shape (N, 1)."""
        s = self.sigmas.index_select(0, t)
        if batch is not None:
            s = s.index_select(0, batch)
        return s.unsqueeze(-1)   # (N, 1)

    def index_to_log_onehot(self, v):
        return index_to_log_onehot(v, self.num_classes)

    def onehot_encode(self, v):
        return F.one_hot(v, self.num_classes).float()

    # ------------------------------------------------------------------ forward marginal
    def q_vt_pred(self, log_v0, t, batch=None):
        """Log of q(v_t | v_0): (1-sigma)*p0 + sigma/K."""
        if batch is None:
            batch = torch.arange(t.shape[0], device=t.device)
        sigma = self._sigma(t, batch)          # (N, 1)
        p0 = log_v0.exp()                      # (N, K)
        q = (1.0 - sigma) * p0 + sigma / self.num_classes
        return torch.log(q + self.eps).clamp_min(-32.0)

    def q_vt_sample(self, log_v0, t, batch=None):
        """Sample from q(v_t | v_0)."""
        if batch is None:
            batch = torch.arange(t.shape[0], device=t.device)
        log_q = self.q_vt_pred(log_v0, t, batch)
        sample = log_sample_categorical(log_q)
        log_sample = index_to_log_onehot(sample, self.num_classes)
        return sample, log_sample

    def add_noise(self, v, time_step, batch=None):
        """Add noise: returns (v_perturbed, log_vt, log_v0)."""
        if batch is None:
            batch = torch.arange(time_step.shape[0], device=time_step.device)
        log_v0 = index_to_log_onehot(v, self.num_classes)
        v_perturbed, log_vt = self.q_vt_sample(log_v0, time_step, batch)
        v_perturbed = torch.where(time_step[batch] == 0, v, v_perturbed)
        log_vt = torch.where(time_step[batch, None] == 0, log_v0, log_vt)
        return v_perturbed, log_vt, log_v0

    # ------------------------------------------------------------------ posterior
    def q_v_posterior(self, log_v0, log_vt, t, batch=None, v0_prob=True):
        """
        Compute log q(v_{t-1} | v_t, p_0) for one rollback step.

        Derivation (using uniform mixing):
          fact1[k] = q(v_t | v_{t-1}=k)  *  p(v_t observed)  [unnorm]
                   = (1 - gamma_t) * p_vt(k) + gamma_t / K
          fact2[k] = q(v_{t-1}=k | p_0)
                   = (1 - sigma_{t-1}) * p_0(k) + sigma_{t-1} / K
          posterior[k] propto fact1[k] * fact2[k]
        """
        if batch is None:
            batch = torch.arange(t.shape[0], device=t.device)

        t_prev = torch.clamp(t - 1, min=0)
        sigma_t    = self._sigma(t,      batch)   # (N, 1)
        sigma_prev = self._sigma(t_prev, batch)   # (N, 1)

        # gamma_t = (sigma_t - sigma_prev) / (1 - sigma_prev)
        gamma_t = ((sigma_t - sigma_prev) / (1.0 - sigma_prev + 1e-12)).clamp(0.0, 1.0)

        p_vt = log_vt.exp()   # (N, K)
        p0   = log_v0.exp()   # (N, K)

        # fact1: q(v_t | v_{t-1}=k) integrated against observed v_t distribution
        fact1 = (1.0 - gamma_t) * p_vt + gamma_t / self.num_classes   # (N, K)
        # fact2: q(v_{t-1}=k | p_0)
        fact2 = (1.0 - sigma_prev) * p0 + sigma_prev / self.num_classes  # (N, K)

        out = torch.log(fact1 + self.eps) + torch.log(fact2 + self.eps)
        out = out - torch.logsumexp(out, dim=-1, keepdim=True)

        # At t=0, posterior collapses to p_0
        t_expand = t[batch].unsqueeze(-1)
        out = torch.where(t_expand == 0, log_v0, out)
        return out

    # ------------------------------------------------------------------ sampling prior
    def sample_init(self, n):
        """Sample from the noise prior (sigma_max state = fully uniform)."""
        init_log = torch.log(self._init_prob_buf + self.eps).clamp_min(-32.0)
        init_log = init_log.unsqueeze(0).expand(n, -1)
        init_types  = log_sample_categorical(init_log)
        init_onehot = self.onehot_encode(init_types)
        log_vt = index_to_log_onehot(init_types, self.num_classes)
        return init_types, init_onehot, log_vt


class CategoricalTransition(nn.Module):
    def __init__(self, betas, num_classes):
        super().__init__()
        self.num_classes = num_classes
        
        alphas  = 1. - betas
        alphas_bar = np.cumprod(alphas, axis=0)
        alphas_bar_prev = np.concatenate([[1.], alphas_bar[:-1]])

        self.betas = to_torch_const(betas)
        self.alphas = to_torch_const(alphas)
        self.alphas_bar = to_torch_const(alphas_bar)
        self.alphas_bar_prev = to_torch_const(alphas_bar_prev)

        log_alphas = np.log(alphas)
        log_alphas_bar = np.cumsum(log_alphas)

        self.log_alphas = to_torch_const(log_alphas)
        self.log_1_min_alphas = to_torch_const(log_1_min_a(log_alphas))
        self.log_alphas_bar = to_torch_const(log_alphas_bar)
        self.log_1_min_alphas_bar = to_torch_const(log_1_min_a(log_alphas_bar))

    def add_noise(self, v, time_step, batch):
        # Vt = a * V0 + (1-a) / K
        log_node_v0 = index_to_log_onehot(v, self.num_classes)
        v_perturbed, log_node_vt = self.q_vt_sample(log_node_v0, time_step, batch)
        v_perturbed = F.one_hot(v_perturbed, self.num_classes).float()
        return v_perturbed, log_node_vt, log_node_v0
    
    def onehot_encode(self, v):
        return F.one_hot(v, self.num_classes).float()

    def q_vt_sample(self, log_v0, t, batch):
        # sample from q(vt | v0)
        log_q_vt_v0 = self.q_vt_pred(log_v0, t, batch)
        sample_class = log_sample_categorical(log_q_vt_v0)
        log_sample = index_to_log_onehot(sample_class, self.num_classes)
        return sample_class, log_sample

    def q_vt_pred(self, log_v0, t, batch):
        # compute q(vt | v0)
        ndim = log_v0.ndim
        log_alpha_bar_t = extract(self.log_alphas_bar, t, batch, ndim=ndim)
        log_1_min_alpha_bar = extract(self.log_1_min_alphas_bar, t, batch, ndim=ndim)

        log_probs = log_add_exp(
            log_v0 + log_alpha_bar_t,
            log_1_min_alpha_bar - np.log(self.num_classes)
        )
        return log_probs

    def q_v_pred_one_timestep(self, log_vt_1, t, batch):
        # q(vt | vt-1)
        ndim = log_vt_1.ndim
        log_alpha_t = extract(self.log_alphas, t, batch, ndim=ndim)
        log_1_min_alpha_t = extract(self.log_1_min_alphas, t, batch, ndim=ndim)
        # alpha_t * vt + (1 - alpha_t) 1 / K
        log_probs = log_add_exp(
            log_vt_1 + log_alpha_t,
            log_1_min_alpha_t - np.log(self.num_classes)
        )
        return log_probs

    def q_v_posterior(self, log_v0, log_vt, t, batch):
        # q(vt-1 | vt, v0) = q(vt | vt-1, x0) * q(vt-1 | x0) / q(vt | x0)
        t_minus_1 = t - 1
        t_minus_1 = torch.where(t_minus_1 < 0, torch.zeros_like(t_minus_1), t_minus_1)  # Remove negative values, will not be used anyway for final decoder

        log_qvtmin_v0 = self.q_vt_pred(log_v0, t_minus_1, batch)  # q(vt-1 | v0)
        # num_axes = (1,) * (len(log_v0.size()) - 1)
        # t_broadcast = t.view(-1, *num_axes) * torch.ones_like(log_v0)
        ndim = log_v0.ndim
        if ndim == 2:
            t_expand = t[batch].unsqueeze(-1)
        elif ndim == 3:
            t_expand = t[batch].unsqueeze(-1).unsqueeze(-1)
        else:
            raise NotImplementedError('ndim not supported')
        log_qvtmin_v0 = torch.where(t_expand == 0, log_v0, log_qvtmin_v0)

        unnormed_logprobs = log_qvtmin_v0 + self.q_v_pred_one_timestep(log_vt, t, batch)
        log_vtmin_given_vt_v0 = unnormed_logprobs - torch.logsumexp(unnormed_logprobs, dim=-1, keepdim=True)
        return log_vtmin_given_vt_v0

    def compute_v_Lt(self, log_v_post_true, log_v_post_pred, log_v0, t, batch):
        kl_v = categorical_kl(log_v_post_true, log_v_post_pred)
        decoder_nll_v = - log_categorical(log_v0, log_v_post_pred)

        ndim = log_v_post_true.ndim
        if ndim == 2:
            mask = (t == 0).float()[batch]
        elif ndim == 3:
            mask = (t == 0).float()[batch].unsqueeze(-1)
        else:
            raise NotImplementedError('ndim not supported')
        loss_v = mask * decoder_nll_v + (1 - mask) * kl_v
        return loss_v
        
    def sample_init(self, n):
        init_log_atom_vt = torch.zeros(n, self.num_classes).to(self.betas.device)
        init_types = log_sample_categorical(init_log_atom_vt)
        init_onehot = self.onehot_encode(init_types)
        log_vt = index_to_log_onehot(init_types, self.num_classes)
        return init_types, init_onehot, log_vt


class GeneralCategoricalTransition(nn.Module):
    def __init__(self, betas, num_classes, init_prob=None):
        super().__init__()
        self.eps = 1e-30
        self.num_classes = num_classes
        if init_prob is None:
            self.init_prob = np.ones(num_classes) / num_classes
        else:
            if isinstance(init_prob, str):
                if init_prob == 'absorb':  # absorb all states into the first one
                    init_prob = 0.01 * np.ones(num_classes)
                    init_prob[0] = 1
                    self.init_prob = init_prob / np.sum(init_prob)
                elif init_prob == 'absorb_pure':  # absorb all states into the first one
                    init_prob = np.zeros(num_classes)
                    init_prob[0] = 1
                    self.init_prob = init_prob / np.sum(init_prob)
                elif init_prob == 'tomask':  # absorb all states into the the mask type (last one)
                    init_prob = 1e-6 * np.ones(num_classes)
                    init_prob[-1] = 1.
                    self.init_prob = init_prob / np.sum(init_prob)
                elif init_prob == 'tomask0.5':
                    init_prob = np.ones(num_classes)
                    init_prob[-1] = 0.5
                    init_prob[:-1] = 0.5 / (num_classes - 1)
                    self.init_prob = init_prob / np.sum(init_prob)
                elif init_prob == 'uniform':
                    self.init_prob = np.ones(num_classes) / num_classes
                else:
                    raise ValueError(f'Unknown init_prob mode: {init_prob}')
            else:
                init_prob = np.asarray(init_prob, dtype=np.float64)
                self.init_prob = init_prob / np.sum(init_prob)
        self.betas = (betas)
        self.num_timesteps = len(betas)
        
        # Construct transition matrices for q(x_t | x_{t-1})
        q_one_step_mats = [self._get_transition_mat(t) for t in range(0, self.num_timesteps)]
        q_one_step_mats = np.stack(q_one_step_mats, axis=0)  # (T, K, K)
        
        # Construct transition matrices for q(x_t | x_0)
        q_mat_t = q_one_step_mats[0]
        q_mats = [q_mat_t]
        for t in range(1, self.num_timesteps):
            q_mat_t = np.tensordot(q_mat_t, q_one_step_mats[t], axes=[[1], [0]])
            q_mats.append(q_mat_t)
        q_mats = np.stack(q_mats, axis=0)
        
        transpopse_q_onestep_mats = np.transpose(q_one_step_mats, axes=[0, 2, 1])
        
        self.q_mats = to_torch_const(q_mats)
        self.transpopse_q_onestep_mats = to_torch_const(transpopse_q_onestep_mats)

        
    def _get_transition_mat(self, t):
        """Computes transition matrix for q(x_t|x_{t-1}).

        Contrary to the band diagonal version, this method constructs a transition
        matrix with uniform probability to all other states.

        Args:
        t: timestep. integer scalar.

        Returns:
        Q_t: transition matrix. shape = (num_classes, num_classes).
        """
        beta_t = self.betas[t]
        if self.init_prob is None:
            mat = np.full(shape=(self.num_classes, self.num_classes),
                        fill_value=beta_t/float(self.num_classes),
                        dtype=np.float64)
            diag_indices = np.diag_indices_from(mat)
            diag_val = 1. - beta_t * (self.num_classes-1.)/self.num_classes
            mat[diag_indices] = diag_val
        else:
            mat = np.repeat(np.expand_dims(self.init_prob, 0), self.num_classes, axis=0)
            mat = beta_t * mat
            mat_diag = np.eye(self.num_classes) * (1. - beta_t)
            mat = mat + mat_diag
        return mat

    def add_noise(self, v, time_step, batch=None):
        """
        v is the type at time 0
        """
        if batch is None:
            batch = torch.arange(time_step.shape[0]).to(time_step)
        # Vt = a * V0 + (1-a) / K
        log_node_v0 = index_to_log_onehot(v, self.num_classes)
        v_perturbed, log_node_vt = self.q_vt_sample(log_node_v0, time_step, batch)
        # v_perturbed = F.one_hot(v_perturbed, self.num_classes).float()
        v_perturbed = torch.where(time_step[batch] == 0, v, v_perturbed)
        log_node_vt = torch.where(time_step[batch, None] == 0, log_node_v0, log_node_vt)
        return v_perturbed, log_node_vt, log_node_v0
    
    def onehot_encode(self, v):
        return F.one_hot(v, self.num_classes).float()
    
    def index_to_log_onehot(self, v):
        return index_to_log_onehot(v, self.num_classes)

    def q_vt_sample(self, log_v0, t, batch=None):
        # sample from q(vt | v0)
        if batch is None:
            batch = torch.arange(t.shape[0]).to(t.device)
        log_q_vt_v0 = self.q_vt_pred(log_v0, t, batch)
        sample_class = log_sample_categorical(log_q_vt_v0)
        log_sample = index_to_log_onehot(sample_class, self.num_classes)
        return sample_class, log_sample

    def q_vt_pred(self, log_v0, t, batch=None):
        # compute q(vt | v0) // actually represent v_{t+1}
        if batch is None:
            batch = torch.arange(t.shape[0]).to(t.device)
        qt_mat = extract(self.q_mats, t, batch, ndim=1)
        # index_class = log_v0.argmax(dim=-1)
        # q_vt = qt_mat[torch.arange(len(index_class)), index_class]
        q_vt = torch.einsum('...i,...ij->...j', log_v0.exp(), qt_mat)
        return torch.log(q_vt + self.eps).clamp_min(-32.)

    # def q_v_pred_one_timestep(self, log_vt_1, t, batch):
    #     # q(vt | vt-1)
    #     ndim = log_vt_1.ndim
    #     log_alpha_t = extract(self.log_alphas, t, batch, ndim=ndim)
    #     log_1_min_alpha_t = extract(self.log_1_min_alphas, t, batch, ndim=ndim)
    #     # alpha_t * vt + (1 - alpha_t) 1 / K
    #     log_probs = log_add_exp(
    #         log_vt_1 + log_alpha_t,
    #         log_1_min_alpha_t - np.log(self.num_classes)
    #     )
    #     return log_probs

    def q_v_posterior(self, log_v0, log_vt, t, batch=None, v0_prob=True):
        # q(vt-1 | vt, v0) = q(vt | vt-1, x0) * q(vt-1 | x0) / q(vt | x0)
        if batch is None:
            batch = torch.arange(t.shape[0]).to(t.device)
        t_minus_1 = t - 1
        t_minus_1 = torch.where(t_minus_1 < 0, torch.zeros_like(t_minus_1), t_minus_1)  # Remove negative values, will not be used anyway for final decoder

        fact1 = extract(self.transpopse_q_onestep_mats, t, batch, ndim=1)
        # class_vt = log_vt.argmax(dim=-1)
        # fact1 = fact1[torch.arange(len(class_vt)), class_vt]
        fact1 = torch.einsum('bj,bjk->bk', torch.exp(log_vt), fact1)  # (batch, N)
        
        if not v0_prob:  # log_v0 is directly transformed to onehot
            fact2 = extract(self.q_mats, t_minus_1, batch, ndim=1)
            class_v0 = log_v0.argmax(dim=-1)
            fact2 = fact2[torch.arange(len(class_v0)), class_v0]
        else:  # log_v0 contains the probability information
            fact2 = extract(self.q_mats, t_minus_1, batch, ndim=1)  # (batch, N, N)
            fact2 = torch.einsum('bj,bjk->bk', torch.exp(log_v0), fact2)  # (batch, N)
        
        ndim = log_v0.ndim
        if ndim == 2:
            t_expand = t[batch].unsqueeze(-1)
        elif ndim == 3:
            t_expand = t[batch].unsqueeze(-1).unsqueeze(-1)
        else:
            raise NotImplementedError('ndim not supported')
        
        out = torch.log(fact1 + self.eps).clamp_min(-32.) + torch.log(fact2 + self.eps).clamp_min(-32.)
        out = out - torch.logsumexp(out, dim=-1, keepdim=True)
        out_t0 = log_v0
        out = torch.where(t_expand == 0, out_t0, out)
        return out

    def compute_v_Lt(self, log_v_post_true, log_v_post_pred, log_v0, t, batch=None):
        if batch is None:
            batch = torch.arange(t.shape[0]).to(t.device)
        kl_v = categorical_kl(log_v_post_true, log_v_post_pred)
        decoder_nll_v = - log_categorical(log_v0, log_v_post_pred)

        ndim = log_v_post_true.ndim
        if ndim == 2:
            mask = (t == 0).float()[batch]
        elif ndim == 3:
            mask = (t == 0).float()[batch].unsqueeze(-1)
        else:
            raise NotImplementedError('ndim not supported')
        loss_v = mask * decoder_nll_v + (1 - mask) * kl_v
        return loss_v
        
    def sample_init(self, n):
        # init_log_atom_vt = torch.zeros(n, self.num_classes).to(self.q_mats.device)
        init_log_atom_vt = torch.log(
            torch.from_numpy(self.init_prob)+self.eps).clamp_min(-32.).to(self.q_mats.device)
        init_log_atom_vt = init_log_atom_vt.unsqueeze(0).repeat(n, 1)
        init_types = log_sample_categorical(init_log_atom_vt)
        init_onehot = self.onehot_encode(init_types)
        log_vt = index_to_log_onehot(init_types, self.num_classes)
        return init_types, init_onehot, log_vt
