# Dataset

This project uses the **Psychological Manipulation Conversations Dataset** from Kaggle.

- Kaggle slug: `tatheerabbas/psychological-manipulation-conversations-dataset`
- Source file used by the completed run: `manipulational_conversation.jsonl`
- Shape: 10,000 records × 21 original fields
- Language: English conversational text
- Prediction target: `manipulation_type`
- Target classes: `charm_flattery`, `direct_coercion`, `gaslighting`, `guilt_tripping`, `love_bombing`, `neutral`, and `passive_aggressive`

The raw dataset is intentionally excluded from this repository. Downloading and use remain subject to the dataset owner's Kaggle terms.

## Kaggle notebook attachment

For the most reliable reproduction:

1. Open the notebook on Kaggle.
2. Choose **Add Input**.
3. Search for `tatheerabbas/psychological-manipulation-conversations-dataset`.
4. Attach the dataset and enable Internet for pretrained resources.

The notebook also supports programmatic access:

```python
import kagglehub

dataset_path = kagglehub.dataset_download(
    "tatheerabbas/psychological-manipulation-conversations-dataset"
)
```

In a Kaggle notebook, KaggleHub handles authentication and attaches the resource to the notebook input area. Outside Kaggle, configure Kaggle authentication according to the official KaggleHub instructions; never commit `kaggle.json` or an API token.

See [Reproducibility](../docs/REPRODUCIBILITY.md) for the full execution procedure.
