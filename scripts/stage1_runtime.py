"""Non-secret node/runtime provenance for each completed GPU sampler."""
import os
import platform
import subprocess
import sys
import torch


def runtime_metadata():
    result=dict(hostname=platform.node(),python_version=sys.version,cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),torch_cuda_version=torch.version.cuda,
                omp_threads=os.environ.get('OMP_NUM_THREADS'),openblas_threads=os.environ.get('OPENBLAS_NUM_THREADS'),mkl_threads=os.environ.get('MKL_NUM_THREADS'))
    try:
        result['gpu_inventory']=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,driver_version','--format=csv,noheader'],text=True,timeout=10).strip().splitlines()
    except (OSError,subprocess.SubprocessError) as exc:
        result['gpu_inventory_error']=str(exc)
    return result
