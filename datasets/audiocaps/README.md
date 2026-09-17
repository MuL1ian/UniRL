# AudioCaps for LTX-2.3 CLAP training

This converter prepares the official human-written captions from
[AudioCaps](https://github.com/cdjkim/audiocaps) for the LTX-2.3 CLAP reward recipe.
AudioCaps was introduced at NAACL 2019 and contains captions for AudioSet clips.

The generated JSONL is a local artifact and must not be committed. This recipe does not
download or consume the source audio: LTX-2.3 generates audio from each caption and CLAP
scores that generated waveform against the same official caption.

## Source and terms

By default, the converter reads the official CSV files at AudioCaps commit
`d004db3ea1b01cf4fd0347dd8d27db90cadc8809`:

- train: 49,838 clips with one caption each
- validation: 495 clips with five captions each (2,475 caption rows)
- test: 975 clips with five captions each (4,875 caption rows)

The upstream repository says its code and dataset are free to use for academic purposes and
asks users to cite the AudioCaps paper. Review the upstream terms before redistributing or
using the data outside that scope.

## Cook

From the repository root:

```bash
python datasets/audiocaps/prepare_audiocaps.py --out-dir data/audiocaps
```

The default manifests contain the full official train split and 64 distinct clips sampled
deterministically from the official validation split. Validation has five captions per clip;
the converter chooses one deterministically so a clip is not counted five times. Pass
`--keep-all-eval-captions --eval-limit 0` to retain every validation caption.

Each row uses the original AudioCaps caption as both the generation prompt and CLAP target:

```json
{
  "prompt": "Multiple clanging and clanking sounds",
  "metadata": {
    "negative_audio_captions": ["... seven captions from other rows ..."],
    "audiocap_id": "58146",
    "source_dataset": "AudioCaps",
    "source_split": "train"
  }
}
```

The CLAP scorer reads the positive text directly from `prompt`; no separate
audio-caption field or hand-written rewrite is required for this dataset.

The deterministic negative captions are diagnostics only. The optimized reward remains the
matched CLAP cosine. They let the run report mismatched cosine, hardest-negative margin, and
caption-retrieval top-1 without imposing a hand-written class vocabulary.

## Train

```bash
DATA_PATH=data/audiocaps/train.jsonl \
EVAL_DATA_PATH=data/audiocaps/eval.jsonl \
bash examples/run_experiment_single_node.sh \
  diffusion/ltx2/ltx2_3_t2av_clap_trainside
```

Train and evaluation use the official AudioCaps train/validation split boundary.
