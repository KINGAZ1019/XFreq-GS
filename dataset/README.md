This directory is a placeholder for user-provided datasets.

Do not commit raw datasets, generated splits, checkpoints, or experiment outputs here.

Expected layout:

```text
dataset/<dataset_name>/
├─ spectrum/
├─ tx_pos.csv
├─ gateway_info.yml
└─ freq.txt
```

Generated split files and initialized point clouds are written to the experiment directory under `logs/<dataset_name>/<exp_name>/dataset_artifacts/`.
