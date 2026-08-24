# Reproducibility Guide

## Reference Environment

The completed notebook reports:

| Component | Reference value |
|---|---|
| Platform | Kaggle Notebook |
| Python | 3.12.13 |
| NumPy | 2.0.2 |
| pandas | 2.3.3 |
| scikit-learn | 1.6.1 |
| TensorFlow | 2.20.0 |
| PyTorch | 2.10.0 with CUDA 12.8 build |
| gensim | 4.4.0 |
| Transformers | 4.57.6 |
| GPU | Tesla T4; reference runtime reported two CUDA devices |
| Random seed | 42 |

The notebook is designed primarily for Kaggle Python 3.12. Package and accelerator images evolve, so future runtimes may report different patch versions and timings.

## Kaggle Setup

1. Sign in to Kaggle and create a new notebook.
2. Open **Session options** and choose a **T4 GPU** accelerator.
3. Enable **Internet**.
4. Add the dataset `tatheerabbas/psychological-manipulation-conversations-dataset` through **Add Input**.
5. Upload [`CSE440_Psychological_Manipulation_Classification.ipynb`](../notebooks/CSE440_Psychological_Manipulation_Classification.ipynb).
6. Restart the session after changing accelerator or package settings.
7. Run all cells in notebook order.

The reference runtime reported two Tesla T4 devices. At least one CUDA-enabled T4-class GPU is required by the notebook's full-project guard; fewer devices or a different GPU can change memory behavior and runtime.

## Internet Requirement

Internet access is required when any of the following are not already attached or cached:

- the Kaggle dataset;
- the selective package updates in the setup cell;
- `glove-wiki-gigaword-50` through gensim-data;
- the `bert-base-uncased` tokenizer and pretrained weights.

If Internet is disabled, attach all resources as Kaggle inputs before execution and update the notebook's search paths as necessary. Do not bypass a failed download by substituting a different dataset or model while claiming comparable results.

## Dataset Download and Attachment

The approved dataset slug is:

```text
tatheerabbas/psychological-manipulation-conversations-dataset
```

The reference run selects:

```text
manipulational_conversation.jsonl
```

Kaggle's **Add Input** interface is preferred because it makes the input version visible in the notebook. The notebook can also use KaggleHub:

```python
import kagglehub

dataset_path = kagglehub.dataset_download(
    "tatheerabbas/psychological-manipulation-conversations-dataset"
)
```

KaggleHub automatically handles authentication and attaches resources when called inside a Kaggle notebook. Outside Kaggle, configure authentication according to the [official KaggleHub repository](https://github.com/Kaggle/kagglehub). Never store or commit `kaggle.json`, API tokens, or session credentials.

After loading, verify these invariants before proceeding:

- shape: 10,000 rows × 21 original columns;
- required fields: `conversation_id`, `manipulation_type`, and `messages`;
- seven exact target labels;
- 10,000 unique conversation identifiers;
- zero malformed conversations.

## Python Environment

The notebook's first setup cell conditionally installs or updates:

```text
kagglehub>=0.3.12,<1
gensim==4.4.0
wordcloud>=1.9.4,<2
transformers>=4.57,<5
accelerate>=1.1,<2
```

Kaggle's GPU image supplies CUDA-enabled TensorFlow and PyTorch. The setup cell does not replace PyTorch and installs TensorFlow only if it is missing, helping preserve compatibility with Kaggle's CUDA stack.

For a separate local environment, create a Python 3.12 virtual environment and install the direct dependencies:

```bash
python -m venv .venv
```

Activate the environment using the command appropriate for the operating system, then run:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Local GPU installation of TensorFlow and PyTorch is platform-specific. Confirm CUDA support independently before starting the full workflow. CPU-only execution is not equivalent to the completed BERT run and may take substantially longer.

## Hardware and Memory Notes

TensorFlow recurrent models and PyTorch BERT run in the same notebook process. The setup therefore:

- enables TensorFlow memory growth before GPU operations;
- uses mixed precision when a TensorFlow GPU is available;
- uses FP16 for BERT training;
- clears TensorFlow sessions and CUDA caches between model families;
- retains only the selected checkpoint inside each BERT configuration directory;
- limits BERT checkpoint history to reduce disk use.

Do not run multiple copies of the notebook in the same GPU session. If an out-of-memory error occurs, restart the kernel to release allocations before retrying; changing batch sizes produces a different configuration and should be documented rather than silently substituted.

## Execution Order

Run cells from top to bottom. The major stages are:

1. environment detection, package checks, seeding, and output paths;
2. dataset discovery and validation;
3. message parsing, metadata exclusion, and exploratory analysis;
4. exact/near-duplicate investigation and template grouping;
5. group-aware train/validation/test split;
6. TF-IDF, Word2Vec, GloVe, and sequence preparation;
7. nine classical tuning runs;
8. eighteen recurrent and bidirectional tuning runs;
9. three BERT tuning runs;
10. validation-based selection;
11. ten frozen test evaluations;
12. model comparison and confusion matrices;
13. ensemble search, validation ablations, and official error analysis;
14. conclusion and requirement checklist;
15. single-text and manual-example diagnostics;
16. output packaging.

Do not skip directly to test-evaluation cells. They depend on in-memory split objects, selected configurations, and model artifacts produced earlier.

## BERT Runtime

The reference run records:

| Configuration | Epochs | Measured training time |
|---|---:|---:|
| `BERT_C1` | 2 | 216.5061 s |
| `BERT_C2` | 3 | 452.5840 s |
| `BERT_C3` | 3 | 383.2008 s |

The three configuration times total approximately 17.54 minutes. Selected BERT test inference takes 2.3020 seconds for 1,454 records in the reference runtime. These are observed timings, not guarantees; Kaggle allocation, GPU count, package versions, caching, and concurrent utilization can change them.

Successful BERT completion should show:

- `BERT execution enabled: True`;
- CUDA device `Tesla T4`;
- 109,487,623 trainable parameters;
- `BERT_C1`, `BERT_C2`, and `BERT_C3` with `completed` status;
- `BERT_C1` selected;
- BERT official test accuracy and macro-F1 both 1.0000.

## Expected Output Directory

Kaggle writes to:

```text
/kaggle/working/CSE440_Project_Outputs
```

The complete run creates several output families:

### Small summaries

- `duplicate_template_summary.csv`
- `split_distribution.csv`
- `representation_summary.csv`
- `tuning_runs.csv`
- `final_model_comparison.csv`
- `per_class_metrics.csv`
- `ensemble_result.csv`
- `ensemble_validation_weight_search.csv`
- `ablation_results.csv`
- `error_analysis_examples.csv`
- `requirement_checklist.csv`
- `manual_example_model_comparison.csv`
- `manual_example_predictions.csv`

### Figures

- `template_leakage_analysis.png`
- `selected_neural_accuracy_curves.png`
- `model_comparison_charts.png`
- `per_class_f1_heatmap.png`
- `all_normalized_confusion_matrices.png`
- `ablation_results.png`

The combined loss figure is intentionally omitted because its BERT panel was not populated: the original plotting field mapping looked for Keras `val_loss`, while the Hugging Face history records `eval_loss`. BERT training completed successfully, and the epoch logs and metrics remain stored in the notebook.

### Large or model-specific artifacts

- classical `.joblib` estimators;
- recurrent `.keras` models;
- selected BERT checkpoint files;
- training-history CSVs;
- validation and test probability arrays;
- confusion-matrix CSVs and classification reports;
- full test-prediction tables;
- Word2Vec vectors and embedding matrices.

The final packaging cell reports `/kaggle/working/CSE440_Project_Outputs.zip` at 428.06 MB. The archive and model artifacts are intentionally excluded from Git.

## Expected Completion Checks

Before accepting a reproduced run, confirm:

- exact duplicate text rows: 2,081;
- rows with nearest similarity ≥ 0.98: 4,858;
- split sizes: 6,938 / 1,608 / 1,454;
- conversation-ID overlap: 0;
- template-group overlap: 0;
- tuning runs completed: 30/30;
- final evaluations completed: 10/10;
- BERT configurations completed: 3/3;
- requirement-checklist rows: all `True`;
- no notebook output of type `error`.

The expected official metrics are recorded in [Results](RESULTS.md) and [`final_model_comparison.csv`](../results/tables/final_model_comparison.csv).

## Benign Runtime Warnings

The completed notebook contains nonfatal warning streams but no Jupyter error outputs:

- **New BERT classifier head:** the seven-class output head is newly initialized because it is task-specific; downstream fine-tuning is expected and completes.
- **PyTorch scalar gather warning:** multi-GPU collection receives scalar losses and temporarily unsqueezes them; training completes.
- **TensorFlow unknown dataset attribute:** a serialized graph contains `use_unbounded_threadpool`; the runtime states that unknown attributes are ignored.
- **TensorFlow retracing warning:** repeated manual predictions trigger retracing notices; diagnostic predictions still complete.
- **pandas future warning:** a future concatenation behavior is announced without changing the recorded run.

Treat a traceback or a notebook output with `output_type: error` differently: stop, diagnose the failure, and do not report incomplete metrics as reproduced results.

## Reproduction Checklist

- [ ] Correct Kaggle dataset attached
- [ ] Internet enabled or pretrained resources attached
- [ ] T4 GPU selected and CUDA visible
- [ ] Notebook executed from the first cell in order
- [ ] All 30 tuning runs completed
- [ ] All 10 official test evaluations completed
- [ ] BERT completed 3/3 and `BERT_C1` selected
- [ ] Group overlap checks equal zero
- [ ] Manual diagnostics kept separate from official metrics
- [ ] Raw data, credentials, checkpoints, and the large archive kept out of Git
