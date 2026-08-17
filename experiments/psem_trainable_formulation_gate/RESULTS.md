# PSEM trainable formulation gate results

> **Stopped experiment:** Read [`EXPERIMENT_STOP_SUMMARY.ko.md`](EXPERIMENT_STOP_SUMMARY.ko.md) for the consolidated interpretation and limitations. These arms did not fine-tune pretrained encoder parameters, no scratch arm was run, and clean 12-arm regeneration was not completed. The results below therefore do not answer fine-tuning versus scratch.

Evidence status: **development-known direction-selection evidence only**.

The three pinned models used the same ten natural continuous meetings, five out-of-fold splits, source-time grid, fixed-lag context, event semantics, duplicate handling, adapter recipe, and evaluation code.

The primary operating point for each arm maximizes the mean F1 across the 100/250/500 ms collars over the complete score range. FE/h does not select the threshold.

## Full-range precision, recall, and F1

| Model | Arm | Macro F1 | P@100 | R@100 | F1@100 | P@250 | R@250 | F1@250 | P@500 | R@500 | F1@500 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| eres2netv2-standard-prepool | A-FROZEN-DIRECT | 0.3734 | 0.2208 | 0.3473 | 0.2699 | 0.3156 | 0.4964 | 0.3859 | 0.3799 | 0.5975 | 0.4645 |
| eres2netv2-standard-prepool | B-TRAINABLE-DIRECT | 0.3832 | 0.2170 | 0.3854 | 0.2777 | 0.3108 | 0.5519 | 0.3976 | 0.3708 | 0.6584 | 0.4744 |
| eres2netv2-standard-prepool | C-FROZEN-STATE | 0.2888 | 0.1622 | 0.1977 | 0.1782 | 0.2671 | 0.3256 | 0.2935 | 0.3593 | 0.4380 | 0.3948 |
| eres2netv2-standard-prepool | D-TRAINABLE-STATE | 0.2889 | 0.1700 | 0.1849 | 0.1771 | 0.2788 | 0.3033 | 0.2905 | 0.3829 | 0.4165 | 0.3990 |
| wavlm-base-plus | A-FROZEN-DIRECT | 0.4159 | 0.2488 | 0.4657 | 0.3243 | 0.3319 | 0.6213 | 0.4327 | 0.3763 | 0.7045 | 0.4906 |
| wavlm-base-plus | B-TRAINABLE-DIRECT | 0.4131 | 0.2478 | 0.4605 | 0.3222 | 0.3303 | 0.6140 | 0.4296 | 0.3748 | 0.6967 | 0.4874 |
| wavlm-base-plus | C-FROZEN-STATE | 0.3023 | 0.2203 | 0.2039 | 0.2118 | 0.3269 | 0.3027 | 0.3143 | 0.3962 | 0.3667 | 0.3809 |
| wavlm-base-plus | D-TRAINABLE-STATE | 0.3051 | 0.2255 | 0.2083 | 0.2165 | 0.3305 | 0.3053 | 0.3174 | 0.3971 | 0.3667 | 0.3813 |
| mhubert-147 | A-FROZEN-DIRECT | 0.4431 | 0.2900 | 0.4421 | 0.3502 | 0.3821 | 0.5826 | 0.4615 | 0.4284 | 0.6532 | 0.5175 |
| mhubert-147 | B-TRAINABLE-DIRECT | 0.4430 | 0.2901 | 0.4419 | 0.3502 | 0.3816 | 0.5813 | 0.4607 | 0.4289 | 0.6534 | 0.5179 |
| mhubert-147 | C-FROZEN-STATE | 0.3438 | 0.2747 | 0.2195 | 0.2440 | 0.4031 | 0.3221 | 0.3581 | 0.4831 | 0.3860 | 0.4291 |
| mhubert-147 | D-TRAINABLE-STATE | 0.3418 | 0.2684 | 0.2137 | 0.2379 | 0.4013 | 0.3195 | 0.3558 | 0.4869 | 0.3877 | 0.4317 |

## FE/h compatibility references

These legacy reference rows annotate the same frontier. They are not a product policy or the basis for selecting the headline threshold.

| Model | Arm | Target FE/h | Recall@250 | Recall@500 | Actual FE/h |
| --- | --- | ---: | ---: | ---: | ---: |
| eres2netv2-standard-prepool | A-FROZEN-DIRECT | 1 | 0.0011 | 0.0011 | 0.4227 |
| eres2netv2-standard-prepool | A-FROZEN-DIRECT | 5 | 0.0061 | 0.0069 | 4.6498 |
| eres2netv2-standard-prepool | A-FROZEN-DIRECT | 10 | 0.0113 | 0.0121 | 9.9337 |
| eres2netv2-standard-prepool | A-FROZEN-DIRECT | 20 | 0.0156 | 0.0165 | 19.8674 |
| eres2netv2-standard-prepool | B-TRAINABLE-DIRECT | 1 | 0.0013 | 0.0015 | 0.8454 |
| eres2netv2-standard-prepool | B-TRAINABLE-DIRECT | 5 | 0.0028 | 0.0032 | 4.6498 |
| eres2netv2-standard-prepool | B-TRAINABLE-DIRECT | 10 | 0.0067 | 0.0076 | 9.7224 |
| eres2netv2-standard-prepool | B-TRAINABLE-DIRECT | 20 | 0.0171 | 0.0184 | 19.8674 |
| eres2netv2-standard-prepool | C-FROZEN-STATE | 1 | 0.0017 | 0.0019 | 0.8454 |
| eres2netv2-standard-prepool | C-FROZEN-STATE | 5 | 0.0063 | 0.0069 | 4.8612 |
| eres2netv2-standard-prepool | C-FROZEN-STATE | 10 | 0.0121 | 0.0132 | 9.9337 |
| eres2netv2-standard-prepool | C-FROZEN-STATE | 20 | 0.0195 | 0.0229 | 19.8674 |
| eres2netv2-standard-prepool | D-TRAINABLE-STATE | 1 | 0.0037 | 0.0039 | 0.8454 |
| eres2netv2-standard-prepool | D-TRAINABLE-STATE | 5 | 0.0093 | 0.0104 | 4.6498 |
| eres2netv2-standard-prepool | D-TRAINABLE-STATE | 10 | 0.0158 | 0.0173 | 9.9337 |
| eres2netv2-standard-prepool | D-TRAINABLE-STATE | 20 | 0.0240 | 0.0268 | 19.2334 |
| wavlm-base-plus | A-FROZEN-DIRECT | 1 | 0.0002 | 0.0002 | 0.4227 |
| wavlm-base-plus | A-FROZEN-DIRECT | 5 | 0.0035 | 0.0037 | 4.8612 |
| wavlm-base-plus | A-FROZEN-DIRECT | 10 | 0.0139 | 0.0145 | 9.9337 |
| wavlm-base-plus | A-FROZEN-DIRECT | 20 | 0.0318 | 0.0331 | 19.8674 |
| wavlm-base-plus | B-TRAINABLE-DIRECT | 1 | 0.0009 | 0.0011 | 0.8454 |
| wavlm-base-plus | B-TRAINABLE-DIRECT | 5 | 0.0043 | 0.0050 | 4.8612 |
| wavlm-base-plus | B-TRAINABLE-DIRECT | 10 | 0.0113 | 0.0126 | 9.5110 |
| wavlm-base-plus | B-TRAINABLE-DIRECT | 20 | 0.0281 | 0.0301 | 19.4447 |
| wavlm-base-plus | C-FROZEN-STATE | 1 | 0.0045 | 0.0045 | 0.4227 |
| wavlm-base-plus | C-FROZEN-STATE | 5 | 0.0117 | 0.0119 | 4.8612 |
| wavlm-base-plus | C-FROZEN-STATE | 10 | 0.0182 | 0.0191 | 9.9337 |
| wavlm-base-plus | C-FROZEN-STATE | 20 | 0.0312 | 0.0331 | 19.8674 |
| wavlm-base-plus | D-TRAINABLE-STATE | 1 | 0.0037 | 0.0037 | 0.6341 |
| wavlm-base-plus | D-TRAINABLE-STATE | 5 | 0.0106 | 0.0110 | 4.8612 |
| wavlm-base-plus | D-TRAINABLE-STATE | 10 | 0.0204 | 0.0216 | 9.7224 |
| wavlm-base-plus | D-TRAINABLE-STATE | 20 | 0.0359 | 0.0385 | 19.8674 |
| mhubert-147 | A-FROZEN-DIRECT | 1 | 0.0017 | 0.0017 | 0.8454 |
| mhubert-147 | A-FROZEN-DIRECT | 5 | 0.0076 | 0.0080 | 4.6498 |
| mhubert-147 | A-FROZEN-DIRECT | 10 | 0.0149 | 0.0158 | 9.7224 |
| mhubert-147 | A-FROZEN-DIRECT | 20 | 0.0303 | 0.0314 | 19.2334 |
| mhubert-147 | B-TRAINABLE-DIRECT | 1 | 0.0006 | 0.0006 | 0.4227 |
| mhubert-147 | B-TRAINABLE-DIRECT | 5 | 0.0071 | 0.0078 | 4.4385 |
| mhubert-147 | B-TRAINABLE-DIRECT | 10 | 0.0115 | 0.0123 | 9.9337 |
| mhubert-147 | B-TRAINABLE-DIRECT | 20 | 0.0264 | 0.0279 | 19.8674 |
| mhubert-147 | C-FROZEN-STATE | 1 | 0.0063 | 0.0063 | 0.8454 |
| mhubert-147 | C-FROZEN-STATE | 5 | 0.0201 | 0.0210 | 4.6498 |
| mhubert-147 | C-FROZEN-STATE | 10 | 0.0316 | 0.0333 | 9.9337 |
| mhubert-147 | C-FROZEN-STATE | 20 | 0.0543 | 0.0591 | 19.8674 |
| mhubert-147 | D-TRAINABLE-STATE | 1 | 0.0061 | 0.0063 | 0.8454 |
| mhubert-147 | D-TRAINABLE-STATE | 5 | 0.0167 | 0.0173 | 4.8612 |
| mhubert-147 | D-TRAINABLE-STATE | 10 | 0.0247 | 0.0255 | 9.7224 |
| mhubert-147 | D-TRAINABLE-STATE | 20 | 0.0496 | 0.0515 | 19.8674 |

## Structured-state diagnostics

| Model | Arm | State macro F1 | Silence R | Singleton R | Overlap R | Decoder relation BAcc | Decoder different R | Adjacent different R | Gap same R | Gap different R |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| eres2netv2-standard-prepool | C-FROZEN-STATE | 0.6961 | 0.9006 | 0.7025 | 0.6923 | 0.7555 | 0.5373 | 0.0686 | 0.2765 | 0.6304 |
| eres2netv2-standard-prepool | D-TRAINABLE-STATE | 0.7001 | 0.9006 | 0.7096 | 0.6924 | 0.7447 | 0.5049 | 0.0196 | 0.2706 | 0.6012 |
| wavlm-base-plus | C-FROZEN-STATE | 0.7122 | 0.9305 | 0.7072 | 0.7273 | 0.8182 | 0.6477 | 0.1176 | 0.0235 | 0.7529 |
| wavlm-base-plus | D-TRAINABLE-STATE | 0.7122 | 0.9293 | 0.7065 | 0.7298 | 0.8210 | 0.6526 | 0.0882 | 0.0412 | 0.7646 |
| mhubert-147 | C-FROZEN-STATE | 0.7265 | 0.9331 | 0.7145 | 0.7696 | 0.8266 | 0.6656 | 0.1569 | 0.1765 | 0.7665 |
| mhubert-147 | D-TRAINABLE-STATE | 0.7369 | 0.9277 | 0.7360 | 0.7596 | 0.8381 | 0.6899 | 0.1471 | 0.0588 | 0.7977 |

## Interpretation

### eres2netv2-standard-prepool

- A→B adaptation under direct supervision: Δmacro-F1 = +0.0098.
- C→D adaptation under structured supervision: Δmacro-F1 = +0.0001.
- A→C current structured-to-event pipeline with frozen evidence: Δmacro-F1 = -0.0846.
- B→D current structured-to-event pipeline with adapted evidence: Δmacro-F1 = -0.0943.
- Best arm for this model: **B-TRAINABLE-DIRECT**.

### wavlm-base-plus

- A→B adaptation under direct supervision: Δmacro-F1 = -0.0028.
- C→D adaptation under structured supervision: Δmacro-F1 = +0.0027.
- A→C current structured-to-event pipeline with frozen evidence: Δmacro-F1 = -0.1135.
- B→D current structured-to-event pipeline with adapted evidence: Δmacro-F1 = -0.1080.
- Best arm for this model: **A-FROZEN-DIRECT**.

### mhubert-147

- A→B adaptation under direct supervision: Δmacro-F1 = -0.0001.
- C→D adaptation under structured supervision: Δmacro-F1 = -0.0020.
- A→C current structured-to-event pipeline with frozen evidence: Δmacro-F1 = -0.0993.
- B→D current structured-to-event pipeline with adapted evidence: Δmacro-F1 = -0.1011.
- Best arm for this model: **A-FROZEN-DIRECT**.

Overall reference baseline: **mhubert-147 / A-FROZEN-DIRECT**.

The current adapter does not provide a consistent full-range F1 improvement. The current R7-B-style structured-to-event pipeline is worse than direct supervision for every model, but the non-random state and relation diagnostics do not establish that structured representation itself failed. Component quality and the multiplicative event projection remain confounded. Close this learning gate after clean reproduction; do not strengthen the adapter or revise structured learning inside issue #72.

The next step is a no-training error decomposition of the existing mHuBERT-A raw predictions. It must distinguish timing/localization, duplicate peaks, remote acoustic false positives, and missing candidate evidence before any next learning experiment is selected.

The recommendation is limited to the next PSEM training stage. It is not a release, multilingual-generalization, or production-readiness claim.
