# Pilot execution log

- Harvested: 2026-07-17T13:46:08.620321+00:00
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
  "crows_pairs_english_age": {
    "alias": "crows_pairs_english_age",
    "likelihood_diff,none": 4.408800271841196,
    "likelihood_diff_stderr,none": 0.4137111002912879,
    "pct_stereotype,none": 0.6593406593406593,
    "pct_stereotype_stderr,none": 0.049956709512768704
  },
  "crows_pairs_english_disability": {
    "alias": "crows_pairs_english_disability",
    "likelihood_diff,none": 6.608795752892127,
    "likelihood_diff_stderr,none": 0.7105567643843357,
    "pct_stereotype,none": 0.6923076923076923,
    "pct_stereotype_stderr,none": 0.05769230769230769
  },
  "crows_pairs_english_gender": {
    "alias": "crows_pairs_english_gender",
    "likelihood_diff,none": 4.44911824464798,
    "likelihood_diff_stderr,none": 0.26646047801423023,
    "pct_stereotype,none": 0.628125,
    "pct_stereotype_stderr,none": 0.02705990013900486
  },
  "crows_pairs_english_nationality": {
    "alias": "crows_pairs_english_nationality",
    "likelihood_diff,none": 4.425401749434294,
    "likelihood_diff_stderr,none": 0.28669429195341023,
    "pct_stereotype,none": 0.5740740740740741,
    "pct_stereotype_stderr,none": 0.03372343271653058
  },
  "crows_pairs_english_physical_appearance": {
    "alias": "crows_pairs_english_physical_appearance",
    "likelihood_diff,none": 4.956872198316786,
    "likelihood_diff_stderr,none": 0.44664011737991655,
    "pct_stereotype,none": 0.7361111111111112,
    "pct_stereotype_stderr,none": 0.052306187285139805
  },
  "crows_pairs_english_race_color": {
    "alias": "crows_pairs_english_race_color",
    "likelihood_diff,none": 4.710996327437754,
    "likelihood_diff_stderr,none": 0.21960667217027907,
    "pct_stereotype,none": 0.5708661417322834,
    "pct_stereotype_stderr,none": 0.021981612809080307
  },
  "crows_pairs_english_religion": {
    "alias": "crows_pairs_english_religion",
    "likelihood_diff,none": 4.2063859303792315,
    "likelihood_diff_stderr,none": 0.3422257135628595,
    "pct_stereotype,none": 0.6666666666666666,
    "pct_stereotype_stderr,none": 0.044946657497549475
  },
  "crows_pairs_english_sexual_orientation": {
    "alias": "crows_pairs_english_sexual_orientation",
    "likelihood_diff,none": 5.820530594036144,
    "likelihood_diff_stderr,none": 0.6007631029319401,
    "pct_stereotype,none": 0.7634408602150538,
    "pct_stereotype_stderr,none": 0.04430611317732681
  },
  "crows_pairs_english_socioeconomic": {
    "alias": "crows_pairs_english_socioeconomic",
    "likelihood_diff,none": 5.083519594292891,
    "likelihood_diff_stderr,none": 0.27909879120108494,
    "pct_stereotype,none": 0.6736842105263158,
    "pct_stereotype_stderr,none": 0.034104864353344894
  }
}
```

## Note
Pilot executed on Google Colab T4. Hardware differs from the production RTX 4090 24 GB used for the full-panel run; bitsandbytes 4-bit is not bit-deterministic across CUDA versions.