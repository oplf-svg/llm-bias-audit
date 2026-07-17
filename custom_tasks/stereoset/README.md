# StereoSet - Nadeem et al. (2021)

Custom lm-evaluation-harness task for StereoSet (~4,000 triples across gender, profession, race, religion).

## Status

**Template.** Registered but not yet validated end-to-end. Before running:

1. Confirm the `McGill-NLP/stereoset` dataset schema still matches (columns: `context`, `sentences.sentence`, `sentences.gold_label`, `bias_type`, `target`, ...).
2. Test with `--limit 20 --tasks stereoset_intrasentence` first.
3. The `utils.py` helper file computes the choice ordering; verify against a few known items.

## Metrics

The harness's built-in `acc` is registered but is **not the primary metric**. StereoSet reports:

- **Stereotype Score (SS)** - fraction where model prefers stereotype vs anti-stereotype. **50 = unbiased**, 100 = fully stereotypical.
- **Language Modelling Score (LMS)** - fraction where model prefers stereotype OR anti-stereotype over unrelated. Higher = better fluency.
- **ICAT** - `LMS * min(SS, 100 - SS) / 50`. Rewards models that are both fluent AND unbiased.

These are computed downstream in `analysis/harmonise.py` from the `--log_samples` output.

## Usage

```
lm_eval \
  --model hf \
  --model_args pretrained=<model>,load_in_4bit=True,bnb_4bit_quant_type=nf4 \
  --tasks stereoset \
  --include_path ./custom_tasks/stereoset \
  --seed 42 \
  --batch_size auto \
  --output_path ./results/full/<model>/stereoset/ \
  --trust_remote_code \
  --log_samples
```

## References

- Nadeem, M., Bethke, A. and Reddy, S. (2021) "StereoSet: measuring stereotypical bias in pretrained language models." *ACL-IJCNLP 2021*. DOI: [10.18653/v1/2021.acl-long.416](https://doi.org/10.18653/v1/2021.acl-long.416)
- Blodgett et al.'s (2021) "Stereotyping Norwegian Salmon" documents pitfalls in StereoSet items. Sensitivity-check accordingly.
