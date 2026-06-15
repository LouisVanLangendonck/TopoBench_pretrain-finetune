#!/bin/bash
# ==============================================================================
# SCRIPT: gpse_backbone_pretrain_sweep.sh
# DESCRIPTION:
#   Hyperparameter sweep for GSPE_backbone across all pretraining methods (BGRL, DGI,
#   GraphCL, GraphMAEv2, VGAE) on ADME and OGB molecular graph datasets.
#
#   Swept parameters per method are taken directly from the reference
#   generate_inductive_experiment_<method>.py scripts (GPS-based experiments),
#   adapted for GPSE backbone and extended with residual_connections.
#
#   CONCURRENCY: Uses "Virtual Slots" — N parallel jobs per GPU.
#   ORDERING:    Datasets → arch params → weight decay → method params → seeds.
#
# ESTIMATED RUNS (3 datasets × 2 hidden × 2 layers × 1 wd × 1 lr × 1 seed = 12 base):
#   BGRL:       12 × 2 (de2) × 2 (df2) × 2 (mom) × 2 (pool)            =  192
#   DGI:        12 × 2 (crpt) × 2 (readout_type)                        =   48
#   GraphCL:    12 × 4 (aug2) × 2 (r2) × 2 (pool)                      =  192
#   GraphMAEv2: 12 × 2 (mr) × 2 (mom) × 2 (res) × 2 (pool) × 2 (dec)  =  384
#   VGAE:       12 × 2 (esr) × 2 (var) × 2 (pool)                      =   96
#   ──────────────────────────────────────────────────────────────────── = 912
#
# Reduce sweep size by commenting out values in the param arrays (Section 3).
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
    # "graph/BBB_Martins"
    # "graph/CYP3A4_Veith"
    # "graph/PROTEINS"
    # "graph/Caco2_Wang"
    # "graph/Clearance_Hepatocyte_AZ"
    # "graph/ogbg-molhiv"
    # "graph/REDDIT-BINARY"
    # "graph/PPBR_AZ"
    # "graph/CYP2C9_Veith"
    # "graph/Clearance_Microsome_AZ"
    # "graph/DD"
    # "graph/ENZYMES"
    # "graph/COLLAB"
    # "graph/IMDB-BINARY"
    # "graph/Solubility_AqSolDB"
)

# --- GSPE_backbone architecture hyperparameters (shared across all methods) ---
gpse_hidden_channels=(128)   # model.feature_encoder.out_channels
gpse_num_layers=(4 6)            # model.backbone.num_layers


# --- Optimizer sweep (shared, from reference scripts) ---
weight_decays=(0) #0.0001        # optimizer.parameters.weight_decay
learning_rates=(0.001) # 0.0005   # optimizer.parameters.lr

# --- Pretraining methods to run (comment out any you want to skip) ---
PRETRAIN_METHODS=(
    "bgrl"
    "dgi"
    "graphcl"
    "graphmaev2"
    "vgae"
)

# --- Seeds ---
DATA_SEEDS=(0)

# --- Fixed arguments applied to every run (from reference scripts) ---
# Note: backbone.heads is GPS-specific and does not apply to GPSE backbone.
FIXED_ARGS=(
    "model=graph/gpse_backbone"
    "model.backbone.dropout=0.0"
    "model.feature_encoder.proj_dropout=0.2"
    "trainer.min_epochs=10"
    "trainer.check_val_every_n_epoch=2"
    "callbacks.early_stopping.patience=15"
)


# ==============================================================================
# SECTION 4: PER-METHOD SWEEP PARAMETERS
#
# Format: "tag|hydra.key|val1 val2 ..."
#   - tag   : short label used in the run name (empty = value basename used)
#   - key   : Hydra override key
#   - vals  : space-separated values to sweep
#
# Multi-key presets: "alias::key=val@@@key=val" — alias → run-name tag,
#   each @@@-piece is a separate Hydra CLI arg, key field is unused ("_").
#
# For reproducibility every non-swept config knob is pinned in *_FIXED below,
# cross-referencing the corresponding configs/pretraining/<method>.yaml defaults.
# ==============================================================================

# ── BGRL ──────────────────────────────────────────────────────────────────────
# Swept: drop_edge_rate_2, drop_feature_rate_2, momentum, readout pooling_type.
# Fixed: rates_1=0.2, residual_connections=true (always on),
#        force_undirected=false, predictor_hidden_dim=64 (from bgrl.yaml).
bgrl_method_SWEEP=(
    "de2|model.backbone_wrapper.drop_edge_rate_2|0.0 0.2"
    "df2|model.backbone_wrapper.drop_feature_rate_2|0.0 0.2"
    "mom|model.backbone_wrapper.momentum|0.99 0.999"
    "pool|model.readout.pooling_type|sum mean"
)
bgrl_FIXED=(
    "model.backbone_wrapper.drop_edge_rate_1=0.2"
    "model.backbone_wrapper.drop_feature_rate_1=0.2"
    "model.backbone_wrapper.force_undirected=false"
    "model.backbone_wrapper.residual_connections=false"
    "model.readout.predictor_hidden_dim=64"
)

# ── DGI ───────────────────────────────────────────────────────────────────────
# Swept: corruption_type, readout.readout_type (graph summary pooling used in
#        the bilinear discriminator — this is the meaningful DGI pooling knob,
#        not pooling_type which is unused in DGI's forward pass).
# Fixed: residual_connections=true (always on), verbose=false,
#        pooling_type=sum, out_channels=1 (from dgi.yaml).
dgi_method_SWEEP=(
    "crpt|model.backbone_wrapper.corruption_type|feature_shuffle graph_diffusion"
    "pool|model.readout.readout_type|sum mean"
)
dgi_FIXED=(
    "model.backbone_wrapper.residual_connections=false"
    "model.backbone_wrapper.verbose=false"
    "model.readout.pooling_type=sum"
    "model.readout.out_channels=1"
)

# ── GraphCL ───────────────────────────────────────────────────────────────────
# Swept: aug2 (all options except none), aug_ratio2, readout pooling_type.
# Fixed: aug1=mask_attr, aug_ratio1=0.2, residual_connections=true,
#        readout_type=mean, mask_attr_strategy=zeros,
#        edge_perturbation_mode=drop_only, subgraph_ratio_meaning=keep,
#        projection_type=linear (from graphcl.yaml).
graphcl_method_SWEEP=(
    "a2|model.backbone_wrapper.aug2|drop_node drop_edge mask_attr subgraph"
    "r2|model.backbone_wrapper.aug_ratio2|0.2 0.5"
    "pool|model.readout.pooling_type|sum mean"
)
graphcl_FIXED=(
    "model.backbone_wrapper.aug1=mask_attr"
    "model.backbone_wrapper.aug_ratio1=0.2"
    "model.backbone_wrapper.residual_connections=false"
    "model.backbone_wrapper.readout_type=mean"
    "model.backbone_wrapper.mask_attr_strategy=zeros"
    "model.backbone_wrapper.edge_perturbation_mode=drop_only"
    "model.backbone_wrapper.subgraph_ratio_meaning=keep"
    "model.readout.projection_type=linear"
)

# ── GraphMAEv2 ────────────────────────────────────────────────────────────────
# Swept: mask_rate, momentum, residual_connections, pooling_type, decoder_type.
# Fixed: replace_rate=0.0, drop_edge_rate=0.0, delayed_ema_epoch=0, lam=1.0,
#        decoder_hidden_dim=256, num_remasking=5, remask_rate=0.5,
#        remask_method=random (from graphmaev2.yaml).
graphmaev2_method_SWEEP=(
    "mr|model.backbone_wrapper.mask_rate|0.3 0.7"
    "mom|model.backbone_wrapper.momentum|0.99 0.996"
    "pool|model.readout.pooling_type|sum mean"
    "dec|model.readout.decoder_type|mlp gcn"
)
graphmaev2_FIXED=(
    "model.backbone_wrapper.replace_rate=0.0"
    "model.backbone_wrapper.residual_connections=false"
    "model.backbone_wrapper.drop_edge_rate=0.0"
    "model.backbone_wrapper.delayed_ema_epoch=0"
    "model.backbone_wrapper.lam=1.0"
    "model.readout.decoder_hidden_dim=256"
    "model.readout.num_remasking=5"
    "model.readout.remask_rate=0.5"
    "model.readout.remask_method=random"
)

# ── VGAE ──────────────────────────────────────────────────────────────────────
# variational=false gives deterministic GAE (z = mu, no reparameterization).
# Swept: edge_sample_ratio, variational, readout pooling_type.
# Fixed: neg_sample_ratio=1.0, sampling_method=sparse, latent_dim=32,
#        residual_connections=false, decoder_type=dot, decoder_hidden_dim=64,
#        out_channels=1 (from vgae.yaml).
vgae_method_SWEEP=(
    "esr|model.backbone_wrapper.edge_sample_ratio|0.2 0.8"
    "var|model.backbone_wrapper.variational|true false"
    "pool|model.readout.pooling_type|sum mean"
)
vgae_FIXED=(
    "model.backbone_wrapper.neg_sample_ratio=1.0"
    "model.backbone_wrapper.sampling_method=sparse"
    "model.backbone_wrapper.latent_dim=32"
    "model.backbone_wrapper.residual_connections=false"
    "model.readout.decoder_type=dot"
    "model.readout.decoder_hidden_dim=64"
    "model.readout.out_channels=1"
)


# ==============================================================================
# SECTION 5: COMBINATION GENERATOR
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
# SECTION 6: COLLECT ALL COMBINATIONS
# ==============================================================================

echo "----------------------------------------------------------"
echo " Generating combinations for all pretraining methods..."
echo "----------------------------------------------------------"

tmp_combos=$(mktemp)
total_runs=0

# Shared sweep dimensions applied to every method
shared_SWEEP=(
    "|dataset|${datasets[*]}"
    "h|model.feature_encoder.out_channels|${gpse_hidden_channels[*]}"
    "L|model.backbone.num_layers|${gpse_num_layers[*]}"
    "wd|optimizer.parameters.weight_decay|${weight_decays[*]}"
    "lr|optimizer.parameters.lr|${learning_rates[*]}"
    "seed|dataset.split_params.data_seed|${DATA_SEEDS[*]}"
)

for method in "${PRETRAIN_METHODS[@]}"; do

    method_SWEEP=()
    method_SWEEP+=("|pretraining|${method}")
    method_SWEEP+=("${shared_SWEEP[@]}")

    case "${method}" in
        bgrl)       method_SWEEP+=("${bgrl_method_SWEEP[@]}") ;;
        dgi)        method_SWEEP+=("${dgi_method_SWEEP[@]}") ;;
        graphcl)    method_SWEEP+=("${graphcl_method_SWEEP[@]}") ;;
        graphmaev2) method_SWEEP+=("${graphmaev2_method_SWEEP[@]}") ;;
        vgae)       method_SWEEP+=("${vgae_method_SWEEP[@]}") ;;
    esac

    method_count=0
    while IFS=";" read -r col1 col2; do
        if [[ "${col1}" == "TOTAL" ]]; then
            method_count=${col2}
            total_runs=$(( total_runs + method_count ))
            printf "  %-12s %5d combinations\n" "[${method}]" "${method_count}"
        else
            echo "${col1};${col2}" >> "${tmp_combos}"
        fi
    done < <(generate_combinations "${method_SWEEP[@]}")

done

echo "----------------------------------------------------------"
echo "► Total runs planned: ${total_runs}"
echo "----------------------------------------------------------"


# ==============================================================================
# SECTION 7: MAIN EXECUTION LOOP
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

    # ── Extract method and dataset for W&B project / per-method fixed args ───
    pretrain_method=""
    dataset_val=""
    for arg in "${DYNAMIC_ARGS_ARRAY[@]}"; do
        if [[ "${arg}" == pretraining=* ]]; then pretrain_method="${arg#*=}"; fi
        if [[ "${arg}" == dataset=*      ]]; then dataset_val=$(basename "${arg#*=}"); fi
    done

    # ── Batch size / max epochs: COLLAB uses smaller values ──────────────────
    if [[ "${dataset_val}" == "COLLAB" ]]; then
        batch_size=8
        max_epochs=40
    else
        batch_size=256
        max_epochs=400
    fi

    # ── Per-method fixed overrides (not swept, not in run name) ──────────────
    method_fixed_args=()
    case "${pretrain_method}" in
        bgrl)       method_fixed_args=("${bgrl_FIXED[@]}") ;;
        dgi)        method_fixed_args=("${dgi_FIXED[@]}") ;;
        graphcl)    method_fixed_args=("${graphcl_FIXED[@]}") ;;
        graphmaev2) method_fixed_args=("${graphmaev2_FIXED[@]}") ;;
        vgae)       method_fixed_args=("${vgae_FIXED[@]}") ;;
    esac

    dynamic_project="${script_name}_${pretrain_method}_${dataset_val}"

    # ── Build and launch ──────────────────────────────────────────────────────
    cmd=(
        "python" "-m" "topobench"
        "${DYNAMIC_ARGS_ARRAY[@]}"
        "${FIXED_ARGS[@]}"
        "${method_fixed_args[@]}"
        "dataset.dataloader_params.batch_size=${batch_size}"
        "trainer.max_epochs=${max_epochs}"
        "trainer.devices=[${current_gpu}]"
        "+logger.wandb.entity=${WANDB_ENTITY}"
        "logger.wandb.project=${dynamic_project}"
    )

    cmd_eval=$(printf '%q ' "${cmd[@]}")
    run_and_log "${cmd_eval% }" "${log_group}" "${run_name}" "${LOG_DIR}" &
    slot_pids[$assigned_slot]=$!

done < "${tmp_combos}"

rm -f "${tmp_combos}"


# ==============================================================================
# SECTION 8: CLEANUP
# ==============================================================================

echo "----------------------------------------------------------"
echo " All ${run_counter} jobs launched."
echo " Waiting for remaining background jobs to finish..."
echo "----------------------------------------------------------"
wait
echo "✔ All runs complete."
echo "📋 Summary: ${LOG_DIR}/_summary.log"
