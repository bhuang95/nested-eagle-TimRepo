# stage1b_bo Failure Analysis (job 54509672)

## Cause of the failure

The job crashed during transfer-learning weight loading from the stage1a checkpoint:

```
File ".../anemoi/training/utils/checkpoint.py", line 106, in transfer_learning_loading
    model._ckpt_model_name_to_index = checkpoint["hyper_parameters"]["data_indices"].name_to_index
AttributeError: 'dict' object has no attribute 'name_to_index'
```

This is **a bug in anemoi-training's `transfer_learning_loading`, not in the config or
checkpoint.** Why:

- In this anemoi version (0.x.post1082), checkpoints store
  `hyper_parameters["data_indices"]` as a **dict keyed by dataset name** — the stage1a
  checkpoint contains `{'data': <IndexCollection>}`. This is the nested/multi-dataset format.
- The normal resume path handles this correctly — `BaseGraphModule.on_load_checkpoint`
  (base.py:385) iterates `data_indices.items()`, and the consumer in `train.py:244` expects a
  dict `{dataset_name: name_to_index}`.
- But `transfer_learning_loading` (checkpoint.py:106) was never updated for that format. It
  still assumed the old single-`IndexCollection` form and called `.name_to_index` directly on
  the dict -> `AttributeError`.

Everything before that worked fine: the shape-mismatch skips in the log
(`latlons_data` 64800 -> 1035360, `latlons_hidden` 10944 -> 40320) are expected and correct —
that is transfer learning from the coarser stage1a grid to the finer stage1b grid, which is
exactly why `transfer_learning: true` / `load_weights_only: true` are set.

## The fix (applied)

Patched `~/.conda/envs/eagle/.../anemoi/training/utils/checkpoint.py` to build the per-dataset
dict, mirroring the already-correct `on_load_checkpoint`:

```python
model._ckpt_model_name_to_index = {
    dataset_name: data_indices.name_to_index
    for dataset_name, data_indices in checkpoint["hyper_parameters"]["data_indices"].items()
}
```

This produces the `{dataset_name: name_to_index}` structure that `train.py:244` then zips
against `self.data_indices.items()` for the variable-consistency check. Module imports cleanly.

Resubmit `submit_stage1b_bo.sh` as-is — no config changes needed.

## Two caveats

1. **This edits a file inside the conda env**, so it will be reverted if anemoi-training is
   reinstalled/upgraded. If the env is shared with collaborators, consider reporting it
   upstream as a transfer-learning bug for nested data_indices.
2. Unrelated warning in the log:
   `UserWarning: Dataset has no tendency statistics! Are you sure you want to use a tendency
   scaler?` — if stage1b_bo.yaml configures a tendency scaler but the dataset lacks tendency
   stats, that may cause a different problem later.
