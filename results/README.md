# Result Tables

The files in [`tables/`](tables/) are compact result tables recovered from the stored displays in the completed notebook. They retain the precision shown in that executed run; no model training was rerun to prepare this repository.

| File | Contents |
|---|---|
| `final_model_comparison.csv` | Frozen test accuracy, F1, timing, and model size for all ten models |
| `tuning_runs.csv` | All 30 completed validation-stage configurations |
| `selected_configurations.csv` | The ten configurations selected by validation macro-F1 |
| `per_class_f1.csv` | Official test F1 for every model and class |
| `ensemble_result.csv` | Validation-selected soft-voting ensemble and improvement flag |
| `ablation_results.csv` | Validation-only controlled ablations |
| `manual_example_model_comparison.csv` | Seven-example diagnostic summary across all models |
| `manual_example_predictions.csv` | Predictions for each of the seven diagnostic conversations |
| `logistic_regression_14_examples.csv` | Separate fourteen-example Logistic Regression diagnostic |
| `duplicate_template_summary.csv` | Exact-text, normalized-text, and near-duplicate investigation |
| `split_distribution.csv` | Per-class train, validation, and test counts |
| `target_class_distribution.csv` | Full-dataset class counts and percentages |
| `representation_summary.csv` | TF-IDF, Word2Vec, and GloVe characteristics |
| `requirement_checklist.csv` | Notebook completion and integrity checks |

Large probability arrays, full prediction dumps, raw split identifiers, serialized models, checkpoints, and the 428.06 MB output archive are deliberately excluded. See [Results](../docs/RESULTS.md) for interpretation.
