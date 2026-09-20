#!/usr/bin/env bash
# Fixed version of run_pickscore_sanity.sh.
#
# Root cause of that run's ~0 grad_norm (confirmed via wandb history: grad_norm
# ~3.7e-5, loss ~-8.5e-7, ratio_std=0, approx_kl=0, clip_fraction=0 -- while
# rollout/advantage_std stayed ~1.0, i.e. the reward signal itself was fine):
# ltx2_3_t2av_trainside.yaml runs with audio_joint_sde=true but never sets
# bundle.config.audio_policy_logp_weight, so it defaults to None. Per
# unirl/models/ltx2/diffusion.py's _combine_modality_logp(), audio_weight=None
# means the video+audio policy log-prob is combined by *element-count*
# weighting: (video_logp*n_video + audio_logp*n_audio) / (n_video+n_audio).
# Audio-latent element counts vastly outnumber this recipe's low-res/short
# video latents, so the combined log-prob -- and therefore the FlowGRPO
# ratio/gradient -- was almost entirely carried by the (unrewarded) audio
# stream, starving gradient to the video LoRA modules that PickScore actually
# scores. This mirrors exactly why the CLAP recipe (ltx2_3_t2av_clap_trainside)
# explicitly sets audio_policy_logp_weight: 1.0 (pure-audio policy, since its
# reward is audio-only) -- ltx2_3_t2av_trainside.yaml simply predates that
# PR's audio_policy_logp_weight feature and was never updated for the
# symmetric video-only case.
#
# Fix applied below: +bundle.config.audio_policy_logp_weight=0.0, i.e. the
# combined log-prob collapses to pure video_logp -- matching what PickScore
# actually rewards. Everything else (checkpoint, dataset, reward, LoRA
# targets, batch size) is unchanged from run_pickscore_sanity.sh.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${REPO_ROOT}"

ENV_DIR="/apdcephfs_fsgm3/share_305110755/hunyuan/boye/envs/leo2-runtime-py312-torch271"
export PATH="${ENV_DIR}/bin:${PATH}"

export PRETRAINED_MODEL="/apdcephfs_fsgm3/share_305110755/hunyuan/boye/datasets/LTX-2.3-Diffusers"

# PickScore + its CLIP processor are already cached here from the leo2 setup.
export HF_HOME="/apdcephfs_gz44/share_305110755/hunyuan/HYvideo/leo2_rl/hf_cache"
export HF_HUB_OFFLINE=1

# Ray otherwise prestarts one Python worker per detected CPU (~376) and the
# resulting module-import storm on CephFS hangs placement-group creation.
export RAY_OVERRIDE_RESOURCES="${RAY_OVERRIDE_RESOURCES:-{\"CPU\": 32}}"

# Pod has no route to api.wandb.ai (confirmed again: DNS resolves, TCP 443
# connect fails; the pod's GIT_PROXY_ADDRESS proxy only allows CONNECT to
# git/github hosts, 404s on api.wandb.ai / huggingface.co) -- log offline,
# sync later with `wandb sync <run-dir>` from a host that has internet.
export WANDB_MODE="${WANDB_MODE:-offline}"
export REPORT_TO_WANDB=true
export WANDB_PROJECT="${WANDB_PROJECT:-leo2_traing}"
export WANDB_RUN_NAME="ltx2_3_t2av_picksore_fixed"
export WANDB_API_KEY="${WANDB_API_KEY:-$(grep -o 'WANDB_API_KEY=.*' /apdcephfs_fsgm3/share_305110755/hunyuan/boye/UniRL/.env | cut -d= -f2-)}"

exec bash "${REPO_ROOT}/examples/run_experiment_single_node.sh" \
  diffusion/ltx2/ltx2_3_t2av_trainside num_rollouts=500 \
  +bundle.config.audio_policy_logp_weight=0.0 "$@"
