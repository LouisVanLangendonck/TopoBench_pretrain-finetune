#!/bin/bash
# ==============================================================================
# SCRIPT: gpse_backbone_supervised_sweep_transductive.sh
# DESCRIPTION:
#   Supervised (pretraining=none) hyperparameter sweep for the GPSE backbone
#   on transductive node-classification datasets.  Serves as a fully-trained
#   baseline to compare against the pretrained variants from
#   gpse_backbone_pretrain_sweep_transductive.sh.
#
#   Swept parameters:
#     hidden_channels : 128, 256, 512   (model.feature_encoder.out_channels)
#     num_layers      : 2, 4, 6         (model.backbone.num_layers)
#     weight_decay    : 0, 0.001        (optimizer.parameters.weight_decay)
#     lr              : 0.001, 0.01     (optimizer.parameters.lr)
#
# ESTIMATED RUNS (4 datasets × 3 hidden × 3 layers × 2 wd × 2 lr × 1 seed):
#   4 × 3 × 3 × 2 × 2 = 144 total runs
#
# CONCURRENCY: Uses "Virtual Slots" — N parallel jobs per GPU.
# All runs log to a single W&B project named after this script.
# ==============================================================================


# ==============================================================================
# SECTION 1: LOGGING & ENVIRONMENT SETUP
# ==============================================================================

script_name="$(basename "${BASH_SOURCE[0]}" .sh)"
log_group="${script_name}"
LOG_DIR="./logs/${log_group}"

# ─── EDIT: set your W&B entity ────────────────────────────────────────────────
WANDB_ENTITY="louis-van-langendonck-universitat-polit-cnica-de-catalunya"
# ──────────────────────────────────────────────────────────────────────────────

echo "=========================================================="
echo " Script  : ${script_name}"
echo " Log dir : ${LOG_DIR}"
echo "=========================================================="

if [ -d "${LOG_DIR}" ]; then rm -r "${LOG_DIR}"; fi
mkdir -p "${LOG_DIR}"

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
export HYDRA_FULL_ERROR=1

find_logging_script() {
    local dir="$1"
    while [[ "${dir}" != "/" ]]; do
        if [[ -f "${dir}/base/logging.sh" ]];         then echo "${dir}/base/logging.sh";         return 0; fi
        if [[ -f "${dir}/scripts/base/logging.sh" ]]; then echo "${dir}/scripts/base/logging.sh"; return 0; fi
        dir="$(dirname "${dir}")"
    done
    return 1
}

LOGGING_PATH=$(find_logging_script "${SCRIPT_DIR}")
if [[ -n "${LOGGING_PATH}" ]]; then
    echo "✔ Logging utils: ${LOGGING_PATH}"
    source "${LOGGING_PATH}"
else
    echo "❌ CRITICAL: Could not locate 'base/logging.sh'."
    exit 1
fi

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1


# ==============================================================================
# SECTION 2: HARDWARE & CONCURRENCY (Auto-Detected)
# ==============================================================================

_gpu_info=$(python3 -c "
import subprocess
try:
    out = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=index,memory.total', '--format=csv,noheader,nounits'],
        text=True
    )
    indices, mem_mb = [], []
    for line in out.strip().splitlines():
        idx, mem = line.split(',')
        indices.append(idx.strip())
        mem_mb.append(int(mem.strip()))
    min_mem_gb = min(mem_mb) / 1024
    if min_mem_gb >= 80:
        jobs = 4
    elif min_mem_gb <= 30:
        jobs = 2
    else:
        jobs = 3
    print(jobs, ' '.join(indices))
except Exception:
    print('2 0')
")
read -r JOBS_PER_GPU _gpu_ids <<< "${_gpu_info}"
read -ra physical_gpus <<< "${_gpu_ids}"

echo "✔ Detected ${#physical_gpus[@]} GPU(s): ${physical_gpus[*]}"
echo "✔ Jobs per GPU: ${JOBS_PER_GPU}"

gpus=()
for gpu in "${physical_gpus[@]}"; do
    for ((i=1; i<=JOBS_PER_GPU; i++)); do gpus+=("${gpu}"); done
done
echo "✔ Total virtual slots: ${#gpus[@]}"

declare -a slot_pids
for i in "${!gpus[@]}"; do slot_pids[$i]=0; done


# ==============================================================================
# SECTION 3: EXPERIMENT PARAMETERS
# ==============================================================================

# --- Datasets ---
datasets=(
    "graph/cocitation_cora"
    "graph/cocitation_pubmed"
    "graph/minesweeper"
    "graph/roman_empire"
)

# --- GPSE backbone architecture hyperparameters ---
gpse_hidden_channels=(128 256 512)   # model.feature_encoder.out_channels
gpse_num_layers=(2 4 6)              # model.backbone.num_layers

# --- Optimizer sweep ---
weight_decays=(0 0.001)              # optimizer.parameters.weight_decay
learning_rates=(0.001 0.01)          # optimizer.parameters.lr

# --- Seeds ---
DATA_SEEDS=(0)

# --- Fixed arguments applied to every run ---
FIXED_ARGS=(
    "model=graph/gpse_backbone"
    "model.backbone.dropout=0.0"
    "model.feature_encoder.proj_dropout=0.2"
    "trainer.max_epochs=400"
    "trainer.min_epochs=10"
    "trainer.check_val_every_n_epoch=2"
    "callbacks.early_stopping.patience=15"
)


# ==============================================================================
# SECTION 4: COMBINATION GENERATOR
# Produces a TOTAL header then one "run_name;arg1 arg2 ..." line per combo.
# ==============================================================================

generate_combinations() {
python3 -c "
import sys, itertools, os

specs = []
for item in sys.argv[1:]:
    parts = item.split('|')
    tag  = parts[0].strip()
    key  = parts[1].strip()
    vals = parts[2].split()
    specs.append({'tag': tag, 'key': key, 'vals': vals})

options      = [[(s['tag'], s['key'], val) for val in s['vals']] for s in specs]
combinations = list(itertools.product(*options))

print(f'TOTAL;{len(combinations)}')

for combo in combinations:
    name_parts = []
    cmd_args   = []
    for (tag, key, val) in combo:
        if '::' in val:
            alias, hydra_val_str = val.split('::', 1)
            clean_val  = alias
            actual_val = hydra_val_str
        else:
            clean_val  = os.path.basename(val)
            actual_val = val

        if tag:
            name_parts.append(f'{tag}{clean_val}')
        else:
            name_parts.append(clean_val)

        if '@@@' in actual_val:
            for part in actual_val.split('@@@'):
                part = part.strip()
                if part:
                    cmd_args.append(part)
        else:
            cmd_args.append(f'{key}={actual_val}')

    run_name = '_'.join(name_parts)
    print(f'{run_name};' + ' '.join(cmd_args))
" "${@}"
}


# ==============================================================================
# SECTION 5: COLLECT ALL COMBINATIONS
# ==============================================================================

# ── Optional flags ────────────────────────────────────────────────────────────
DEBUG_MODE=false
for arg in "$@"; do
    [[ "${arg}" == "--debug" ]] && DEBUG_MODE=true
done
if [[ "${DEBUG_MODE}" == "true" ]]; then
    echo "🐛 DEBUG MODE: will run first combination in foreground and show output live."
fi
# ──────────────────────────────────────────────────────────────────────────────

echo "----------------------------------------------------------"
echo " Generating combinations..."
echo "----------------------------------------------------------"

tmp_combos=$(mktemp)
total_runs=0

# pretraining=none is the first dimension — mirrors how pretraining=<method>
# is the first dimension in gpse_backbone_pretrain_sweep_transductive.sh,
# ensuring it is always the first CLI override passed to topobench.
sweep_dims=(
    "|pretraining|none"
    "|dataset|${datasets[*]}"
    "h|model.feature_encoder.out_channels|${gpse_hidden_channels[*]}"
    "L|model.backbone.num_layers|${gpse_num_layers[*]}"
    "wd|optimizer.parameters.weight_decay|${weight_decays[*]}"
    "lr|optimizer.parameters.lr|${learning_rates[*]}"
    "seed|dataset.split_params.data_seed|${DATA_SEEDS[*]}"
)

while IFS=";" read -r col1 col2; do
    if [[ "${col1}" == "TOTAL" ]]; then
        total_runs=${col2}
        printf "  %5d combinations\n" "${total_runs}"
    else
        echo "${col1};${col2}" >> "${tmp_combos}"
    fi
done < <(generate_combinations "${sweep_dims[@]}")

echo "----------------------------------------------------------"
echo "► Total runs planned: ${total_runs}"
echo "----------------------------------------------------------"


# ==============================================================================
# SECTION 6: MAIN EXECUTION LOOP
# ==============================================================================

run_counter=0
one_percent_step=1
if [[ "${total_runs}" -gt 0 ]]; then
    one_percent_step=$(( total_runs / 100 ))
fi
if [[ "${one_percent_step}" -eq 0 ]]; then one_percent_step=1; fi

echo "► Reporting progress every ${one_percent_step} runs (1%)"
echo "=========================================================="

while IFS=";" read -r run_name dynamic_args_str; do
    [[ -z "${run_name}" ]] && continue

    (( run_counter++ ))
    if (( run_counter % one_percent_step == 0 )); then
        percent=$(( (run_counter * 100) / total_runs ))
        echo "📊 Progress: ${percent}% completed (${run_counter} / ${total_runs} launched)"
    fi

    # ── Find a free GPU slot ──────────────────────────────────────────────────
    assigned_slot=-1
    while [[ "${assigned_slot}" -eq -1 ]]; do
        for i in "${!gpus[@]}"; do
            pid="${slot_pids[$i]}"
            if [[ "${pid}" -eq 0 ]] || ! kill -0 "${pid}" 2>/dev/null; then
                assigned_slot=${i}
                break
            fi
        done
        if [[ "${assigned_slot}" -eq -1 ]]; then
            wait -n
        fi
    done

    current_gpu="${gpus[$assigned_slot]}"

    # ── Parse dynamic args ────────────────────────────────────────────────────
    IFS=$' \t\n' read -ra DYNAMIC_ARGS_ARRAY <<< "${dynamic_args_str}"

    # ── Extract swept values for explicit W&B logging ─────────────────────────
    varied_params=()
    for arg in "${DYNAMIC_ARGS_ARRAY[@]}"; do
        case "${arg}" in
            pretraining=*)                        varied_params+=("++varied_param_pretraining=${arg#*=}") ;;
            dataset=*)                            varied_params+=("++varied_param_dataset=$(basename "${arg#*=}")") ;;
            model.feature_encoder.out_channels=*) varied_params+=("++varied_param_hidden_channels=${arg#*=}") ;;
            model.backbone.num_layers=*)          varied_params+=("++varied_param_num_layers=${arg#*=}") ;;
            optimizer.parameters.weight_decay=*)  varied_params+=("++varied_param_weight_decay=${arg#*=}") ;;
            optimizer.parameters.lr=*)            varied_params+=("++varied_param_lr=${arg#*=}") ;;
            dataset.split_params.data_seed=*)     varied_params+=("++varied_param_data_seed=${arg#*=}") ;;
        esac
    done

    # ── Build and launch ──────────────────────────────────────────────────────
    cmd=(
        "python" "-m" "topobench"
        "${DYNAMIC_ARGS_ARRAY[@]}"
        "${FIXED_ARGS[@]}"
        "${varied_params[@]}"
        "trainer.devices=[${current_gpu}]"
        "+logger.wandb.entity=${WANDB_ENTITY}"
        "logger.wandb.project=${script_name}"
    )

    cmd_eval=$(printf '%q ' "${cmd[@]}")

    if [[ "${DEBUG_MODE}" == "true" ]]; then
        echo ""
        echo "🐛 DEBUG: running FIRST combination in foreground (GPU ${current_gpu})"
        echo "🐛 CMD: ${cmd_eval% }"
        echo "────────────────────────────────────────────────────"
        eval "${cmd_eval% }"
        echo "────────────────────────────────────────────────────"
        echo "🐛 Exit code: $?"
        echo "🐛 Stopping after first combination in debug mode."
        rm -f "${tmp_combos}"
        exit 0
    fi

    run_and_log "${cmd_eval% }" "${log_group}" "${run_name}" "${LOG_DIR}" &
    slot_pids[$assigned_slot]=$!

done < "${tmp_combos}"

rm -f "${tmp_combos}"


# ==============================================================================
# SECTION 7: CLEANUP
# ==============================================================================

echo "----------------------------------------------------------"
echo " All ${run_counter} jobs launched."
echo " Waiting for remaining background jobs to finish..."
echo "----------------------------------------------------------"
wait
echo "✔ All runs complete."
echo "📋 Summary: ${LOG_DIR}/_summary.log"

# ── Show tails of the first few failed runs ───────────────────────────────────
if [[ -f "${LOG_DIR}/_summary.log" ]]; then
    n_ok=$(grep -c "✔ OK" "${LOG_DIR}/_summary.log" 2>/dev/null || echo 0)
    n_fail=$(grep -c "✗ FAIL" "${LOG_DIR}/_summary.log" 2>/dev/null || echo 0)
    echo "📊 Results: ${n_ok} OK, ${n_fail} FAILED"
    if [[ "${n_fail}" -gt 0 ]]; then
        echo ""
        echo "══════════════════════════════════════════════════════"
        echo " Tailing the first 3 FAILED run logs for diagnosis:"
        echo "══════════════════════════════════════════════════════"
        grep "✗ FAIL" "${LOG_DIR}/_summary.log" | head -3 | while IFS= read -r line; do
            run_name_raw="${line##*| }"
            safe_name="${run_name_raw//\//_}"
            fail_log="${LOG_DIR}/${safe_name}.log"
            echo ""
            echo "--- ${run_name_raw} ---"
            if [[ -f "${fail_log}" ]]; then
                tail -40 "${fail_log}"
            else
                echo "(log file not found: ${fail_log})"
            fi
        done
    fi
fi
