# Results

## Evaluation Boundaries

Four result sources are reported separately:

1. **Validation results** select one configuration per model family and tune ensemble weights.
2. **Frozen official test results** evaluate the ten selected configurations on 1,454 held-out records.
3. **Seven manual examples** probe every selected model with one newly written example per class.
4. **Fourteen Logistic Regression examples** provide a second, separate diagnostic for the recommended practical model.

Manual-example scores are not combined with official test metrics.

## Completion Summary

- Classical configurations completed: 9/9
- Recurrent and bidirectional configurations completed: 18/18
- BERT configurations completed: 3/3
- Total tuning configurations completed: 30/30
- Selected-model test evaluations completed: 10/10
- Selected BERT configuration: `BERT_C1`
- Stored notebook error outputs: 0

## Selected Validation Configurations

| Model | Configuration | Representation | Validation accuracy | Validation macro-F1 | Epochs / best | Training (s) |
|---|---|---|---:|---:|---:|---:|
| Random Forest | `RF_C3` | Character TF-IDF + SVD | 1.0000 | 1.0000 | 0 / 0 | 10.2700 |
| Logistic Regression | `LR_C1` | TF-IDF | 1.0000 | 1.0000 | 0 / 0 | 1.4822 |
| Naive Bayes | `NB_C2` | TF-IDF | 1.0000 | 1.0000 | 0 / 0 | 0.3801 |
| SimpleRNN | `SRNN_C3` | Word2Vec | 0.9726 | 0.9705 | 6 / 6 | 74.7046 |
| GRU | `GRU_C3` | Word2Vec | 0.9981 | 0.9979 | 6 / 6 | 7.8851 |
| LSTM | `LSTM_C3` | Word2Vec | 0.9994 | 0.9993 | 6 / 5 | 8.0176 |
| Bidirectional SimpleRNN | `BSRNN_C3` | Word2Vec | 0.9938 | 0.9937 | 6 / 6 | 108.8325 |
| Bidirectional GRU | `BGRU_C1` | Word2Vec | 0.9981 | 0.9983 | 4 / 4 | 5.3663 |
| Bidirectional LSTM | `BLSTM_C3` | Word2Vec | 1.0000 | 1.0000 | 6 / 6 | 11.0800 |
| BERT Base | `BERT_C1` | BERT WordPiece | 0.9994 | 0.9995 | 2 / 1 | 216.5061 |

The complete 30-row validation record is available in [`tuning_runs.csv`](../results/tables/tuning_runs.csv). All recorded failure-reason fields are blank.

### BERT tuning

All three BERT configurations have validation macro-F1 0.9995 at the displayed precision. Their measured training times are:

| Configuration | Epochs | Validation macro-F1 | Training (s) |
|---|---:|---:|---:|
| `BERT_C1` | 2 | 0.9995 | 216.5061 |
| `BERT_C2` | 3 | 0.9995 | 452.5840 |
| `BERT_C3` | 3 | 0.9995 | 383.2008 |

`BERT_C1` is selected by the validation ranking and deterministic ordering. The three measured configuration runs total approximately 17.54 minutes.

## Frozen Official Test Results

| Model | Accuracy | Macro-F1 | Weighted-F1 | Training (s) | Inference (s) | Size (MB) |
|---|---:|---:|---:|---:|---:|---:|
| Random Forest | 1.0000 | 1.0000 | 1.0000 | 10.2700 | 0.5547 | 30.5229 |
| Logistic Regression | 1.0000 | 1.0000 | 1.0000 | 1.4822 | 0.1139 | 0.2502 |
| Naive Bayes | 1.0000 | 1.0000 | 1.0000 | 0.3801 | 0.0847 | 0.4031 |
| SimpleRNN | 0.9732 | 0.9692 | 0.9729 | 74.7046 | 0.9380 | 1.0122 |
| GRU | 0.9979 | 0.9977 | 0.9979 | 7.8851 | 0.3630 | 1.1776 |
| LSTM | 0.9993 | 0.9992 | 0.9993 | 8.0176 | 0.4190 | 1.2578 |
| Bidirectional SimpleRNN | 0.9966 | 0.9960 | 0.9966 | 108.8325 | 1.3698 | 1.1083 |
| Bidirectional GRU | 0.9972 | 0.9968 | 0.9973 | 5.3663 | 0.4949 | 0.6388 |
| Bidirectional LSTM | 0.9993 | 0.9992 | 0.9993 | 11.0800 | 0.5925 | 1.5995 |
| BERT Base | 1.0000 | 1.0000 | 1.0000 | 216.5061 | 2.3020 | 418.5920 |

Random Forest, Logistic Regression, Naive Bayes, and BERT Base tie at 1.0000 accuracy, macro-F1, and weighted-F1. No one of these four is the unique best model.

![Official test performance and efficiency](../assets/figures/model_comparison_charts.png)

## Per-Class Test F1

| Model | Charm / flattery | Direct coercion | Gaslighting | Guilt tripping | Love bombing | Neutral | Passive aggressive |
|---|---:|---:|---:|---:|---:|---:|---:|
| Random Forest | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Logistic Regression | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Naive Bayes | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| SimpleRNN | 0.9237 | 0.9700 | 0.9838 | 0.9810 | 0.9802 | 0.9730 | 0.9728 |
| GRU | 0.9924 | 0.9972 | 1.0000 | 0.9973 | 0.9966 | 1.0000 | 1.0000 |
| LSTM | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9975 | 0.9967 |
| Bidirectional SimpleRNN | 0.9962 | 0.9890 | 1.0000 | 1.0000 | 1.0000 | 0.9899 | 0.9967 |
| Bidirectional GRU | 1.0000 | 0.9945 | 1.0000 | 1.0000 | 1.0000 | 0.9900 | 0.9933 |
| Bidirectional LSTM | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9975 | 0.9967 |
| BERT Base | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

![Per-class official test F1](../assets/figures/per_class_f1_heatmap.png)

The class supports in the official test set are 131 charm/flattery, 182 direct coercion, 308 gaslighting, 185 guilt tripping, 299 love bombing, 198 neutral, and 151 passive aggressive records.

## Cost Comparison: Logistic Regression and BERT

Both models obtain 1.0000 accuracy and macro-F1 on the official frozen test set, but their measured costs differ substantially:

| Measure | Logistic Regression | BERT Base |
|---|---:|---:|
| Serialized size | 0.2502 MB | 418.5920 MB |
| Selected-configuration training | 1.4822 s | 216.5061 s |
| Full-test inference | 0.1139 s | 2.3020 s |
| Official macro-F1 | 1.0000 | 1.0000 |

Logistic Regression is therefore the practical deployment recommendation for this benchmark. This recommendation concerns resource efficiency under equal measured test performance; it does not claim that the linear model will be universally superior on natural conversations.

## Ensemble Result

| Components | Weights | Validation macro-F1 | Test accuracy | Test macro-F1 | Improvement |
|---|---|---:|---:|---:|---|
| Random Forest, Bidirectional LSTM, BERT Base | 0.2, 0.6, 0.2 | 1.0000 | 1.0000 | 1.0000 | False |

The weights are selected from validation probabilities. Because four individual models already score 1.0000 on the frozen test set, the ensemble provides no genuine macro-F1 improvement.

## Validation Ablations

| Ablation family | Variant A | Macro-F1 A | Variant B | Macro-F1 B |
|---|---|---:|---|---:|
| Speaker markers | With markers | 1.0000 | Without markers | 1.0000 |
| Preprocessing | Minimal | 1.0000 | Normalized | 1.0000 |
| Average embedding | Word2Vec | 0.9992 | GloVe | 0.9625 |
| SimpleRNN direction | Unidirectional | 0.5669 | Bidirectional | 0.9714 |
| GRU direction | Unidirectional | 0.9604 | Bidirectional | 0.9983 |
| LSTM direction | Unidirectional | 0.9929 | Bidirectional | 0.9987 |
| Split diagnostic | Group-aware | 1.0000 | Ordinary stratified | 1.0000 |

These comparisons use validation results only. The ordinary stratified diagnostic's perfect score does not replace the group-aware policy; it reinforces that this synthetic, lexically regular dataset can be easy under multiple split policies.

![Validation-only ablation comparison](../assets/figures/ablation_results.png)

## Seven-Example Manual Diagnostic

| Model | Correct | Total | Accuracy | Macro-F1 | Mean confidence |
|---|---:|---:|---:|---:|---:|
| BERT Base | 6 | 7 | 0.8571 | 0.8095 | 0.9444 |
| Logistic Regression | 6 | 7 | 0.8571 | 0.8095 | 0.4615 |
| Naive Bayes | 6 | 7 | 0.8571 | 0.8095 | 0.8848 |
| Random Forest | 5 | 7 | 0.7143 | 0.6190 | 0.3638 |
| GRU | 5 | 7 | 0.7143 | 0.6190 | 0.8014 |
| Bidirectional LSTM | 5 | 7 | 0.7143 | 0.6190 | 0.7367 |
| LSTM | 4 | 7 | 0.5714 | 0.5238 | 0.8383 |
| Bidirectional SimpleRNN | 4 | 7 | 0.5714 | 0.4762 | 0.6837 |
| Bidirectional GRU | 4 | 7 | 0.5714 | 0.4524 | 0.7570 |
| SimpleRNN | 4 | 7 | 0.5714 | 0.4286 | 0.5981 |

BERT Base, Logistic Regression, and Naive Bayes each classify 6/7 examples correctly. Random Forest classifies 5/7. The passive-aggressive example is the most consistent difficulty: BERT predicts neutral with 0.9621 confidence, while Logistic Regression and Naive Bayes predict direct coercion.

## Fourteen-Example Logistic Regression Diagnostic

The separate Logistic Regression set contains two examples per class. It scores 11/14, or 78.57%. The three misses are:

- one gaslighting example predicted as passive aggressive;
- one guilt-tripping example predicted as direct coercion;
- one passive-aggressive example predicted as direct coercion.

The row-level results are in [`logistic_regression_14_examples.csv`](../results/tables/logistic_regression_14_examples.csv). This diagnostic must not be averaged with the seven-example comparison or the official test set.

## Official Error Analysis

The deterministic tie-selected Random Forest row is used as the leading-model representative and has 0 errors among 1,454 official test records. This does not make Random Forest a unique leader; three other models have the same official aggregate result.

SimpleRNN is the weakest official model:

- 39 errors among 1,454 records;
- mean confidence 0.6260 on incorrect predictions;
- 11 manipulation examples predicted as neutral;
- 0 neutral examples predicted as manipulation.

Its most frequent confusion pairs are:

| True class | Predicted class | Count |
|---|---|---:|
| `charm_flattery` | `love_bombing` | 10 |
| `passive_aggressive` | `neutral` | 7 |
| `charm_flattery` | `gaslighting` | 5 |
| `guilt_tripping` | `direct_coercion` | 4 |
| `direct_coercion` | `neutral` | 4 |
| `gaslighting` | `direct_coercion` | 3 |

SimpleRNN accuracy by length bucket is 0.9412 for short, 0.9742 for medium, and 0.9775 for long conversations. Its context accuracy is lowest for family conversations at 0.9669.

The complete figure is available as [normalized confusion matrices for all ten models](../assets/figures/all_normalized_confusion_matrices.png).

## Main Interpretation

The perfect official scores should be interpreted in light of the data-generating process, not as evidence that the task is solved in natural settings:

- The dataset is synthetic.
- Classes contain strong lexical cues.
- TF-IDF captures those cues directly and efficiently.
- Recurring templates remain prevalent, even though template components are kept disjoint across partitions.
- BERT requires substantially more computation and storage without improving the official score.
- The manual examples produce clearly lower results, indicating weaker generalization outside the synthetic test distribution.

The recommended practical model is Logistic Regression for this benchmark. Any real-world use would require independent human-reviewed data, broader demographic and conversational coverage, calibration, robustness testing, and a carefully limited decision-support role.
