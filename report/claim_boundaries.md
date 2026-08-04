# Claim boundaries

## Supported by verified artifacts

- Deterministic, disjoint BoolQ subsets contain 800 training, 200
  calibration, and 400 test inputs.
- Six label-blind evidence conditions were generated for every test input.
- Qwen2.5-1.5B-Instruct baseline inference completed for 2,400 rows.
- Three independently seeded LoRA adapters completed explicit PyTorch
  training; losses were finite, gradients reached all trainable parameters,
  weights changed, and checkpoints reloaded within tolerance.
- Temperature scaling used only calibration labels and preserved predicted
  classes.
- A three-adapter ensemble and a Laplace-approximated Bayesian binary linear
  head over frozen representations completed evaluation.
- Saved predictions produced traceable metrics, risk–coverage tables, paired
  changes, expressed-confidence summaries, and eight verified figures.
- The highest observed overall accuracy was
  73.88% for calibrated LoRA
  seed 2; this is a descriptive result for the fixed study protocol.

## Required qualifications

- LoRA is parameter-efficient adaptation, not full-model training.
- Only the linear prediction head is Laplace approximated; the transformer
  is not Bayesian.
- LoRA ensemble members are independently seeded models, not posterior
  samples. Their entropy Jensen gap is mutual-information-style.
- Generated numerical confidence is verbalized output, not a statistical
  probability.
- Expressed-confidence divergence is reported only for parser-valid rows;
  invalid rows remain missing.
- Lexical evidence removal is a deterministic overlap proxy, minimum lexical
  overlap does not prove irrelevance, and templated negation does not prove
  contradiction of the labelled answer.
- Evidence conditions are categorical and must not be interpreted as a
  monotonic severity scale.
- Lower ECE alone does not establish better overall reliability.
- Test results are post-hoc descriptive evidence, not tuning or model
  selection data.

## Unsupported claims

- State-of-the-art or large-scale LLM training.
- A fully Bayesian LLM or exact Bayesian inference.
- Statistically significant improvements or population-level generalization.
- Robustness to arbitrary distribution shift or adversarial attacks.
- Semantically guaranteed evidence removal, distraction, or contradiction.
- Publication, acceptance, or peer review.

## Permitted description

This work may be described as a **reproducible application-stage empirical
study of confidence alignment and uncertainty quantification under controlled
evidence perturbations**, implemented with PyTorch, Transformers, PEFT,
temperature scaling, a three-member LoRA ensemble, and a
Laplace-approximated Bayesian prediction head over frozen LLM
representations.
