#!/usr/bin/env bash
# Sanity check: does LTX-2.3 + FlowGRPO train cleanly with a reward that has
# nothing to do with CLAP/audio (VideoPickScore, video-only, same reward class
# leo2's own recipe uses)? Same checkpoint/pod/env as the CLAP repro
# (ltx2_3_t2av_clap_trainside) -- only the reward + LoRA target modules +
# frame count differ, per the repo's own ltx2_3_t2av_trainside.yaml recipe.
# Goal: isolate whether "recipe doesn't seem to work great" is CLAP-specific
# or something about the LTX-2.3 T2AV joint-SDE training path in general.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${REPO_ROOT}"

ENV_DIR="/apdcephfs_fsgm3/share_305110755/hunyuan/boye/envs/leo2-runtime-py312-torch271"
export PATH="${ENV_DIR}/bin:${PATH}"

export PRETRAINED_MODEL="/apdcephfs_fsgm3/share_305110755/hunyuan/boye/datasets/LTX-2.3-Diffusers"

# PickScore + its CLIP processor are already cached here from the leo2 setup
# (this recipe uses the exact same yuvalkirstain/PickScore_v1 +
# laion/CLIP-ViT-H-14-laion2B-s32B-b79K pair as leo2's VideoPickScore reward).
export HF_HOME="/apdcephfs_gz44/share_305110755/hunyuan/HYvideo/leo2_rl/hf_cache"
export HF_HUB_OFFLINE=1

# Same fix as the CLAP repro and the original leo2 smoke test: Ray otherwise
# prestarts one Python worker per detected CPU (~376) and the resulting
# module-import storm on CephFS hangs placement-group creation for minutes.
export RAY_OVERRIDE_RESOURCES="${RAY_OVERRIDE_RESOURCES:-{\"CPU\": 32}}"

# The pod has no route to api.wandb.ai (confirmed while running the CLAP
# repro) -- log offline, sync later with `wandb sync <run-dir>` from a host
# that has internet.
export WANDB_MODE="${WANDB_MODE:-offline}"
export REPORT_TO_WANDB=true
export WANDB_PROJECT="${WANDB_PROJECT:-leo2_traing}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-ltx2_pickscore_sanity_boye_$(date +%m%d_%H%M)}"
export WANDB_API_KEY="${WANDB_API_KEY:-$(grep -o 'WANDB_API_KEY=.*' /apdcephfs_fsgm3/share_305110755/hunyuan/boye/UniRL/.env | cut -d= -f2-)}"

exec bash "${REPO_ROOT}/examples/run_experiment_single_node.sh" \
    diffusion/ltx2/ltx2_3_t2av_trainside num_rollouts=100 "$@"
