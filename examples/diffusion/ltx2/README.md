# LTX-2.3 CLAP audio training

`ltx2_3_t2av_clap_audiocaps1024_300.yaml` runs 300 optimizer steps: 150 rollouts,
with two disjoint optimizer updates per rollout. Each rollout contains eight
prompts and eight samples per prompt (64 generations). It starts from the
LTX-2.3 dev checkpoint; `load_dir: null` prevents resuming the earlier audio run.

## Data

The prepared data live under `data/audiocaps_clap_1024_128_seed42/`:

- `train.jsonl`: 1,024 captions selected with seed 42 from the existing full
  AudioCaps training JSONL. Captions and YouTube IDs are unique within this set.
- `eval.jsonl`: 128 official validation clips, one caption per clip, selected
  with seed 43 using `datasets/audiocaps/prepare_audiocaps.py` selection logic.
- `manifest.json`: source revision, selection seeds, counts, and overlap checks.

Training captions and YouTube IDs do not overlap the evaluation set. Original
captions, IDs, and source metadata are retained. Each row has seven distractor
captions for retrieval diagnostics. This uses broader AudioCaps prompts,
including speech and mixed events; it is not the earlier 16-caption overfit probe.

## Recipe and evaluation

- 30 denoising steps, CFG 3, 512x768, 121 frames at 24 fps (about five seconds).
- Audio policy log-probability weight 1; existing audio LoRA targets, rank 32,
  alpha 256, learning rate 1e-4, BF16 trainable masters and forward.
- Two SDE transitions sampled from indices 15 through 26, eta 0.3. Siblings
  share initial video/audio noise and receive independent SDE noise.
- Reward is matched CLAP cosine only. Retrieval margin and top-1 remain
  diagnostic metrics; their values do not enter the optimization reward.
- Baseline evaluation at rollout 0, then every ten rollouts (20 optimizer
  steps), on 128 fixed prompts with two fixed-seed samples each, eta 0.
- Checkpoints every 25 rollouts (50 optimizer steps), including the final step.

Track `eval/reward_matched_cosine` and `eval/reward_retrieval_top1`, and listen
to `eval/generated_videos` across evaluations. New total reward excludes the
old retrieval-margin term, so compare the common component metrics across runs.
These are candidate settings; the earlier runs did not validate their convergence.

## Launch on the existing pod

```bash
cd /apdcephfs_fsgm3/share_305110755/hunyuan/boye/workspace/UniRL
export PATH="/apdcephfs_fsgm3/share_305110755/hunyuan/boye/envs/leo2-runtime-py312-torch271/bin:$PATH"
export PRETRAINED_MODEL=/apdcephfs_fsgm3/share_305110755/hunyuan/boye/datasets/LTX-2.3-Diffusers
export HF_HUB_CACHE=/apdcephfs_fsgm3/share_305110755/hunyuan/boye/datasets/hf_cache
export HF_HUB_OFFLINE=1
export RAY_OVERRIDE_RESOURCES='{"CPU": 32}'
export WANDB_MODE=online
export WANDB_PROJECT=leo2_traing
export WANDB_ENTITY=boyeniu-the-university-of-sydney
python -c 'import torch, av; assert torch.cuda.is_available()'
bash examples/run_experiment_single_node.sh diffusion/ltx2/ltx2_3_t2av_clap_audiocaps1024_300
```

Export `WANDB_API_KEY` from the existing `.env` without printing it. PyAV is
installed in the runtime environment so the saved MP4 files include audio.
Set `LTX_AUDIO_SAVE_DIR` for a separate checkpoint directory on repeated runs.
Use `num_rollouts=0` for baseline evaluation only. `num_rollouts=300` would
perform 600 optimizer steps, not 300.

The current background launcher is outside the repository at
`/apdcephfs_fsgm3/share_305110755/hunyuan/boye/Leo_outputs/ltx2_audio300_setup/launch_audio300.py`.
It exports credentials, tries online mode, falls back to offline on W&B
initialization failure, and keeps logs/checkpoints under `Leo_outputs/ltx2_audio300/`.
The companion `monitor_sync.py` runs on the networked host and synchronizes
an offline run every 20 optimizer steps plus completion. Online runs sync
continuously; evaluation remains every 20 optimizer steps in either mode.

## Runtime compatibility

PyTorch 2.7.1 FSDP2 requires one original parameter dtype per shard group.
Using FP32 LoRA masters with frozen BF16 base weights fails on the first
forward. This recipe therefore uses BF16 masters; mixed precision still
reduces gradients in FP32. The initial FP32-master attempt completed zero updates.

## Training without evaluation

`ltx2_3_t2av_clap_audiocaps1024_300_noeval.yaml` starts a fresh 300-update
run with `eval_interval: 0` and no evaluation data path. Baseline and periodic
evaluation are both disabled. Training reward, audio/video previews, and
checkpoints are retained. All sampling and optimization settings inherit the
validated BF16 recipe above. At the first measured rollout time of 1,293 s,
150 rollouts take approximately 54 hours, excluding loading/checkpoint overhead.
This is a single-batch estimate, not a throughput guarantee.

## CPS CLAP-only recipes

The CPS recipes adapt the sampling and optimizer settings from
[verl-omni's LTX-2.3 recipe](https://github.com/verl-project/verl-omni/blob/main/examples/flowgrpo_trainer/ltx2/run_ltx2_3_t2av_lora.sh)
to UniRL trainside FSDP with a single CLAP reward. Both recipes instantiate
`CLAPRewardScorer` directly and need no additional reward model or package.

| Setting | Small batch | High batch |
| --- | --- | --- |
| YAML | `ltx2_3_t2av_verl_cps_clap_only_300.yaml` | `ltx2_3_t2av_verl_cps_clap_only_highbatch32_300.yaml` |
| Prompts x samples | 8 x 8 = 64 per rollout | 32 x 8 = 256 per rollout |
| Global optimizer batch | 32 | 128 |
| Training micro-batch per GPU | 2 | 4 |
| Accumulation micros per update per GPU | 2 | 4 |
| Rollout forward batch per GPU | 2 | 2 |

Common settings:

- VidProM/DanceGRPO prompt list: seed-42 shuffle, first 1,024 held out, 48,976
  training prompts. The source and SHA256 are in `data/vidprom_verl_reference/manifest.json`.
- 24 steps, CFG 4, 256x384 pixels, 81 frames at 24 fps, independent initial noise.
- CPS noise 0.8, three noncontiguous indices from `[0, 10)`, seed `42 + rollout_id`.
- Rank 64 / alpha 128 LoRA on the video, audio, cross-modal attention, and FFN modules.
- AdamW LR 3e-4, weight decay 1e-4, betas .9/.999, epsilon 1e-8; constant LR.
- FlowGRPO clip 1e-4, advantage clip 5, batch-wide advantage std, no reference KL.
- `audio_policy_logp_weight: null`: joint video/audio element-weighted policy score.
  CLAP-only describes the reward; both modalities remain in the policy and LoRA.
- 150 rollouts / 300 optimizer updates. Baseline and periodic evaluation are disabled.
- BF16 model, master parameters, and trajectory; FP32 log-probabilities.

`rollout/reward_mean` equals `rollout/reward_matched_cosine_mean`. Older runs
wrapped CLAP inside a composite backend and named this component
`rollout/reward_clap_mean`; the direct scorer uses the matched-cosine name.
The CLAP scoring formula and weight remain unchanged.

The high-batch run initially used training micro-batch 8 and ran out of memory
in the gradient-enabled replay forward. Micro-batch 4 completed the first
rollout with the same global optimizer batch of 128. It still requires roughly
91 GiB at the observed GPU memory peak, so do not infer spare capacity from
model-loading or generation-only measurements.

### Differences from the reference implementation

- Uses in-process FSDP on eight GPUs instead of TP=2 vLLM-Omni rollout. The
  high-batch global optimizer batch matches the reference; micro-batch 4 fits
  this runtime, while the reference sets micro-batch 8.
- Uses matched CLAP cosine alone. There is no retrieval-margin contribution.
- UniRL recomputes its frozen old-policy anchor with training micro-batches.
  Per-sample noise keys differ from the reference; sparse SDE index selection matches.
- CFG is computed in velocity space under BF16 autocast instead of the
  reference adapter's mathematically equivalent x0-space FP32 expression.
- UniRL uses population std for the batch-wide advantage denominator; the
  reference uses sample std. The scaling difference is about 0.8% for 64
  samples and 0.2% for 256 samples.
- Evaluation is disabled. A rising training reward does not by itself prove
  held-out audio quality or rule out reward exploitation.

### Launch

Set `PRETRAINED_MODEL`, `HF_HUB_CACHE`, `WANDB_API_KEY`, `WANDB_PROJECT`, and
`WANDB_ENTITY` in the existing runtime environment. `av` is needed for audio
in logged MP4 files. Then select one recipe:

```bash
bash examples/run_experiment_single_node.sh diffusion/ltx2/ltx2_3_t2av_verl_cps_clap_only_300
bash examples/run_experiment_single_node.sh diffusion/ltx2/ltx2_3_t2av_verl_cps_clap_only_highbatch32_300
```

The small recipe uses `LTX_VERL_SAVE_DIR`; the high-batch recipe uses
`LTX_CLAP_HIGHBATCH_SAVE_DIR`. Use separate directories for independent runs.
Existing running processes retain the code and reward objects loaded at startup;
this cleanup applies on the next launch and does not restart active runs.

## AudioCaps with normalized event-grounded CLAP

`ltx2_3_t2av_verl_cps_clap_audiocaps_events_300.yaml` inherits the small CPS
recipe above: 8 prompts x 8 samples, global optimizer batch 32, training
micro-batch 2, rollout forward batch 2, and 150 rollouts / 300 updates.
Sampling, LoRA, optimizer, joint AV policy, and sparse CPS indices are unchanged.
It starts from the base model and disables baseline and periodic evaluation.

The input is the original AudioCaps caption, without a visual prompt wrapper.
For example, `A woman talks nearby as water pours` has the official AudioSet
labels `Water tap, faucet` and `Speech`. Join labels by YouTube ID and segment
start time with the converter:

```bash
python datasets/audiocaps/prepare_audiocaps.py \
  --source /path/to/audiocaps-csvs \
  --out-dir data/audiocaps_events \
  --audioset-metadata-dir /path/to/audioset-metadata \
  --eval-limit 64
bash examples/run_experiment_single_node.sh diffusion/ltx2/ltx2_3_t2av_verl_cps_clap_audiocaps_events_300
```

The prepared train split contains 49,838 unique segments. The 64 validation
records are available for later checks; this recipe never evaluates them.
`LTX_AUDIOCAPS_DATA_PATH` overrides the training JSONL and
`LTX_AUDIOCAPS_SAVE_DIR` chooses a separate adapter checkpoint directory.

CLAP receives mono 48 kHz audio normalized toward -20 dBFS RMS, with gain
limited to a peak of 0.95. The train reward is matched-caption cosine plus
the minimum cosine over that clip's AudioSet event labels, each weighted 1.
Both terms use the same CLAP model; ImageBind, AST, and retrieval-margin
rewards are absent. Normalization affects reward input, not saved audio.

Track `rollout/reward_matched_cosine_mean` and
`rollout/reward_event_coverage_min_cosine_mean` alongside `rollout/reward_mean`.
The sum has a different scale from the original CLAP-only reward; changes in
dataset and normalization also prevent an apples-to-apples comparison of
matched cosine with the old VidProM run. Training reward alone does not
establish held-out performance.

This branch includes upstream commits `a53cdf7` (CLAP normalization/events),
`a883c6f` (video FPS metadata), and `6121875` (standard AudioCaps recipe tuning).
The FPS integration uses `primitive_metadata["video"]["fps"]` and
`MediaPreview.video_fps`. The standard recipe's denoising/evaluation settings
do not override this experiment's inherited small CPS configuration.
