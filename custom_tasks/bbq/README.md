# BBQ - Bias Benchmark for QA (Parrish et al., 2022)

Custom lm-evaluation-harness task for the full BBQ dataset (11 categories, ~58,000 items).

## Status

**Template.** Registered but not yet validated end-to-end. Before running:

1. Confirm the `heegyu/bbq` dataset schema still matches (columns: `context`, `question`, `ans0`, `ans1`, `ans2`, `label`, `question_polarity`, `context_condition`, `category`).
2. Test with `--limit 10` first.
3. Duplicate `task_bbq_age.yaml` into 10 more files (`task_bbq_disability_status.yaml`, `task_bbq_gender_identity.yaml`, etc.) - each pointing at the corresponding `dataset_kwargs.name`.

## Categories

- Age
- Disability_status
- Gender_identity
- Nationality
- Physical_appearance
- Race_ethnicity
- Race_x_gender
- Race_x_SES
- Religion
- SES
- Sexual_orientation

## Scoring

`acc` (accuracy) is the built-in metric. The **BBQ bias score** (Parrish et al., 2022 eq. 1) is computed downstream in `analysis/harmonise.py` from the per-sample log:

    bias_score = (n_biased_answers - n_counter-biased_answers) / n_non_unknown_answers

Higher = more biased in the stereotypical direction. Reported alongside accuracy in the interim/dissertation write-ups.

## Usage

```
lm_eval \
  --model hf \
  --model_args pretrained=<model>,load_in_4bit=True,bnb_4bit_quant_type=nf4 \
  --tasks bbq_full \
  --include_path ./custom_tasks/bbq \
  --seed 42 \
  --batch_size auto \
  --output_path ./results/full/<model>/bbq/ \
  --trust_remote_code \
  --log_samples
```

## References

- Parrish, A. et al. (2022) "BBQ: a hand-built bias benchmark for question answering." *Findings of ACL 2022*. DOI: [10.18653/v1/2022.findings-acl.165](https://doi.org/10.18653/v1/2022.findings-acl.165)
