#!/usr/bin/env bash
# Repro run of PR #475 (feat(ltx2): add CLAP-only T2AV reward recipe) on our
# own H20 pod, using the local LTX-2.3-Diffusers checkpoint and locally
# prepared AudioCaps captions + CLAP weights (no network access assumed on
# the training pod itself; both were pre-fetched from a node with internet).
#
# Goal: confirm the recipe trains (reward goes up, no crashes) on our infra
# with zero changes to the recipe itself, before adapting it to score with
# our own audio-pipeline reward instead of CLAP.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${REPO_ROOT}"

ENV_DIR="/apdcephfs_fsgm3/share_305110755/hunyuan/boye/envs/leo2-runtime-py312-torch271"
export PATH="${ENV_DIR}/bin:${PATH}"

export PRETRAINED_MODEL="/apdcephfs_fsgm3/share_305110755/hunyuan/boye/datasets/LTX-2.3-Diffusers"
export DATA_PATH="${REPO_ROOT}/data/audiocaps/train.jsonl"
export EVAL_DATA_PATH="${REPO_ROOT}/data/audiocaps/eval.jsonl"

# CLAP weights were pre-downloaded via snapshot_download(cache_dir=...) into a
# plain HF cache layout -- HF_HUB_CACHE (not HF_HOME) is the right knob for
# that layout. Leave HF_HUB_OFFLINE off in case something else needs a
# network touch; everything this recipe needs is already local.
export HF_HUB_CACHE="/apdcephfs_fsgm3/share_305110755/hunyuan/boye/datasets/hf_cache"
# The GPU pod has no internet route. Without this, transformers' from_pretrained
# still issues a HEAD request per candidate file (adapter_config.json,
# model.safetensors, ...) before falling back to cache, and each failed HEAD
# retries 5x with exponential backoff (~30s) -- multiplied across every
# candidate file and every one of the 8 reward-backend workers. Force offline
# resolution straight to the local snapshot instead.
export HF_HUB_OFFLINE=1

# Same fix as the leo2 smoke test: without this, Ray detects the pod's full
# CPU count (~376) and prestarts that many Python workers at ray.init() time,
# and the resulting concurrent module-import storm on CephFS hangs
# placement-group creation for many minutes.
export RAY_OVERRIDE_RESOURCES="${RAY_OVERRIDE_RESOURCES:-{\"CPU\": 32}}"

# The training pod has no route to api.wandb.ai (confirmed: direct connection
# times out, and the pod's git-only HTTP proxy 404s on CONNECT to it). Log
# offline so wandb.init() doesn't block/timeout, and sync later with
# `wandb sync <run-dir>` from a host that does have internet (the wandb run
# dir lives on shared CephFS, so any such host can see it).
export WANDB_MODE="${WANDB_MODE:-offline}"
export REPORT_TO_WANDB=true
export WANDB_PROJECT="${WANDB_PROJECT:-unirl-ltx2.3-t2av-clap}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-ltx2_clap_repro_boye_$(date +%m%d_%H%M)}"
# shellcheck disable=SC1091
WANDB_API_KEY_VALUE="$(grep -o 'WANDB_API_KEY=.*' /apdcephfs_fsgm3/share_305110755/hunyuan/boye/UniRL/.env | cut -d= -f2-)"
export WANDB_API_KEY="${WANDB_API_KEY:-${WANDB_API_KEY_VALUE}}"

exec bash "${REPO_ROOT}/examples/run_experiment_single_node.sh" \
    diffusion/ltx2/ltx2_3_t2av_clap_trainside "$@"
