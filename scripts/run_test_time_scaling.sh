#!/usr/bin/env bash
# Run through SSH inside an existing Slurm allocation. Logs/results persist.
set -euo pipefail
tts_root=${1:?usage: bash scripts/run_test_time_scaling.sh OUTPUT_ROOT [NUM_POCKETS]}
tts_pockets=${2:-20}
tts_envroot=/shared/healthinfolab/phz24002/anaconda3/envs
tts_python="$tts_envroot/PocketXMol/bin/python"
tts_eval_python="$tts_envroot/reed_molecule/bin/python"
tts_vina_python="$tts_envroot/pxm_vina/bin/python"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export LD_LIBRARY_PATH="$tts_envroot/PocketXMol/lib:${LD_LIBRARY_PATH:-}"
mkdir -p "$tts_root"

sample() {
    local label=$1 steps=$2 repeats=$3 config=$4
    "$tts_python" scripts/sample_test_time_scaling.py --outdir "$tts_root/$label" \
        --steps "$steps" --repeats "$repeats" --num_pockets "$tts_pockets" \
        --batch_size 4 --seed 2026 --config_task "$config" --max_refine_rounds 3 \
        > "$tts_root/$label.sample.log" 2>&1
}
evaluate() {
    local label=$1
    shift
    "$tts_eval_python" scripts/evaluate_test_time_scaling.py --run_dir "$tts_root/$label" \
        --best_of "$@" > "$tts_root/$label.geometry.log" 2>&1
    local tick=$SECONDS
    "$tts_vina_python" evaluate/evaluate_vina_sdf.py --sdf_dir "$tts_root/$label/SDF" \
        --gen_info "$tts_root/$label/gen_info.csv" --protein_dir data/csd/files/proteins \
        --out_csv "$tts_root/$label/vina.csv" --mode score_only --n_workers 2 --quiet \
        > "$tts_root/$label.vina.log" 2>&1
    echo "$((SECONDS - tick))" > "$tts_root/$label/vina_wall_seconds.txt"
    "$tts_eval_python" scripts/evaluate_test_time_scaling.py --run_dir "$tts_root/$label" \
        --best_of "$@" --reuse > "$tts_root/$label.summary.log" 2>&1
}
sample steps100_candidates4 100 4 configs/sample/test/sbdd_csd/simple.yml
evaluate steps100_candidates4 1 2 4
sample steps200 200 1 configs/sample/test/sbdd_csd/simple.yml
evaluate steps200 1
sample steps400 400 1 configs/sample/test/sbdd_csd/simple.yml
evaluate steps400 1
sample native_refine3 100 1 configs/sample/test/sbdd_csd/base.yml
evaluate native_refine3 1
"$tts_eval_python" scripts/evaluate_tts_energy.py --run_dir "$tts_root/steps100_candidates4" \
    --selections best_of_1_confidence best_of_2_confidence best_of_2_geometry_confidence \
    best_of_2_vina_oracle best_of_4_confidence best_of_4_geometry_confidence best_of_4_vina_oracle \
    > "$tts_root/energy.log" 2>&1
for label in steps200 steps400 native_refine3; do
    "$tts_eval_python" scripts/evaluate_tts_energy.py --run_dir "$tts_root/$label" \
        --selections best_of_1_confidence > "$tts_root/$label.energy.log" 2>&1
done
"$tts_eval_python" scripts/summarize_test_time_scaling.py --root "$tts_root"
touch "$tts_root/COMPLETED"
