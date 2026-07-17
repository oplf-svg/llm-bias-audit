# Pilot execution log

- Harvested: 2026-07-17T12:20:01.891395+00:00
- Model: `microsoft/Phi-3-mini-4k-instruct`
- Tasks: crows_pairs_english
- Seed: 42
- Quantisation: load_in_4bit=True, bnb_4bit_quant_type=nf4
- Decoding: greedy (default)

## Hardware
- GPU: Tesla T4, 15360 MiB, 580.82.07
- CUDA runtime: 12.1
- cuDNN: 90100

## Software (all pinned)
- `torch` : `2.4.0`
- `torchvision` : `0.19.0`
- `transformers` : `4.44.2`
- `accelerate` : `0.34.2`
- `bitsandbytes` : `0.43.3`
- `datasets` : `2.20.0`
- `huggingface_hub` : `0.24.6`
- `sentencepiece` : `0.2.0`
- `protobuf` : `5.28.0`
- `numpy` : `1.26.4`
- `lm_eval` : `0.4.7`

## Scores
```json
{
  "crows_pairs_english": {
    "alias": "crows_pairs_english",
    "likelihood_diff,none": 4.778258427737651,
    "likelihood_diff_stderr,none": 0.11419739218718132,
    "pct_stereotype,none": 0.629695885509839,
    "pct_stereotype_stderr,none": 0.01179526464594631
  }
}
```

## Note
Pilot executed on Google Colab T4. Hardware differs from the production RTX 4090 24 GB used for the full-panel run; bitsandbytes 4-bit is not bit-deterministic across CUDA versions.