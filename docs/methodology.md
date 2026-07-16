# Methodology summary

## Design

Quantitative, comparative, positivist. Secondary data only (three published benchmark corpora + one hand-built probe). Cross-sectional snapshot of the 2024-2026 open-weight LLM cohort.

## Panel

Four-to-six open-weight LLMs in the 3B-13B parameter range: Phi-3-mini (3.8B), Mistral-7B-v0.3, Qwen2-7B, Llama-3.1-8B, plus one 2025-2026 release to be selected during pilot.

## Testing protocol

Every model is tested identically. The harness runs each benchmark's canonical scoring routine:

- **BBQ:** multiple-choice QA accuracy in ambiguous and disambiguated contexts + the BBQ bias score.
- **StereoSet:** stereotype score (SS) + language-modelling score (LMS) from likelihood comparisons across stereotype / anti-stereotype / unrelated triples.
- **CrowS-Pairs:** pair-preference rate over stereotypical vs. less-stereotypical sentence pairs.
- **Custom probe (O6):** same likelihood-comparison protocol as CrowS-Pairs, reused directly.

All models share the same seed, greedy decoding, quantisation and GPU class, so score differences reflect model behaviour rather than configuration drift.

## Analysis

- **Score harmonisation.** z-standardise within each benchmark before aggregation.
- **RQ1 - cross-benchmark agreement.** Spearman rho + Kendall tau, with 10 000-resample bootstrap 95% CIs.
- **RQ2 - per-category divergence.** Paired Wilcoxon signed-rank + Friedman across all benchmarks, with rank-biserial effect sizes and Benjamini-Hochberg correction for multiple comparisons.
- **RQ3 - compliance mapping.** Interpret findings against Regulation (EU) 2024/1689 Article 53 and HM Government (2023).

## Ethics

Module-level Low-Risk Secondary Data Science ethics approval. No human participants. Benchmark items retain their original licences (BBQ CC-BY-SA 4.0, StereoSet MIT, CrowS-Pairs academic use).
