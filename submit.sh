#!/bin/bash
# Submit a SLURM job to HPC from the local SSHFS-mounted directory.
# Usage: ./submit.sh <script.sh> [extra sbatch args...]
# Example: ./submit.sh dehazing-diffusion/joint_diffusion/slurm_train.sh
#          ./submit.sh dehazing-diffusion/joint_diffusion/slurm_test.sh --partition=agpu06

HPC_HOST="pinnable-jump"
LOCAL_ROOT="/home/ptthang/f2f_ldm_hpc"
HPC_ROOT="/scrfs/storage/tp030/home/f2f_ldm"

if [ -z "$1" ]; then
    echo "Usage: $0 <slurm_script.sh> [extra sbatch args...]"
    echo ""
    echo "Available SLURM scripts:"
    find "$LOCAL_ROOT" -name 'slurm_*.sh' -printf '  %P\n' | sort
    exit 1
fi

SCRIPT="$1"
shift

# Resolve to absolute local path if relative
if [[ "$SCRIPT" != /* ]]; then
    SCRIPT="$(cd "$LOCAL_ROOT" && realpath --relative-to="$LOCAL_ROOT" "$SCRIPT" 2>/dev/null || echo "$SCRIPT")"
fi

# Strip the local root prefix if present
SCRIPT="${SCRIPT#$LOCAL_ROOT/}"

HPC_SCRIPT="$HPC_ROOT/$SCRIPT"

echo "Submitting: $HPC_SCRIPT"
echo "HPC host:   $HPC_HOST"
[ $# -gt 0 ] && echo "Extra args: $@"
echo "---"

ssh "$HPC_HOST" "sbatch $@ '$HPC_SCRIPT'"
