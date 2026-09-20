# LTX-2 / LTX-2.3

## Gotchas

- `LTX2VerlIndexScheduler` matches verl-omni's three noncontiguous SDE indices
  from `[0, 10)`, using `torch.randperm` and seed `42 + rollout_id`. UniRL
  rollout IDs are zero-based; verl-omni's global training steps are one-based.
- For the reference CPS experiment, `audio_policy_logp_weight: null` combines
  per-element video/audio means by their element counts, equivalent to the
  reference concatenating both streams before reducing its transition score.
- Video previews must carry the pipeline's 24 fps in `primitive_metadata`
  through `MediaPreview`; the logger's legacy 8 fps fallback otherwise stretches
  81 frames to 10.125 seconds while the generated audio remains about 3.3 seconds.
