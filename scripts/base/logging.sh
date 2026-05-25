#!/bin/bash
# ==============================================================================
# scripts/base/logging.sh
# Logging utilities for experiment sweep scripts.
# Source this file to get the run_and_log function.
# ==============================================================================

# run_and_log <cmd_str> <log_group> <run_name> <log_dir>
#
# Evaluates <cmd_str> in the current shell, writing stdout+stderr to a per-run
# log file under <log_dir> and appending a one-liner to the master summary.
# Returns the exit code of the command so the caller can check success.
run_and_log() {
    local cmd_str="$1"
    local log_group="$2"
    local run_name="$3"
    local log_dir="$4"

    # Sanitize run name for use as a filename (replace / with _)
    local safe_name="${run_name//\//_}"
    local log_file="${log_dir}/${safe_name}.log"
    local master_log="${log_dir}/_summary.log"

    # Per-run header
    {
        echo "======================================================"
        echo "Run:     ${run_name}"
        echo "Group:   ${log_group}"
        echo "Started: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "Command: ${cmd_str}"
        echo "======================================================"
    } > "${log_file}"

    # Execute
    eval "${cmd_str}" >> "${log_file}" 2>&1
    local exit_code=$?

    # Per-run footer
    {
        echo "======================================================"
        echo "Finished:  $(date '+%Y-%m-%d %H:%M:%S')"
        echo "Exit code: ${exit_code}"
        echo "======================================================"
    } >> "${log_file}"

    # Append one-liner to master summary (file-level lock via >> which is atomic on Linux)
    if [[ ${exit_code} -eq 0 ]]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✔ OK   | ${run_name}" >> "${master_log}"
    else
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✗ FAIL | exit=${exit_code} | ${run_name}" >> "${master_log}"
    fi

    return ${exit_code}
}
