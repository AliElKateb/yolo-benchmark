# TinyTrain — Selective Filter Freezing

TinyTrain freezes the least important convolutional filters before training, so only the critical ~10–50% of parameters get updated. Adapted from the [TinyTrain paper](https://arxiv.org/abs/2304.09044) for ultralytics YOLOv5/v8.

## Config

```yaml
tiny_train:
  enabled: true
  imp_samples: 256       # images used for importance estimation
  imp_batch_size: 6      # batch size during estimation
  freeze_portion: 0.5    # fraction of filters to freeze
```

## Algorithm

1. **Deep-copy** the `DetectionModel` (`self.model`, not `self.model.model` which is the raw `Sequential` — the `DetectionModel`'s custom `forward()` handles YOLO's skip connections so `Concat` modules receive `list[Tensor]`). Force all params to `requires_grad_(True)`, move to target device.

2. **Importance estimation** — run `imp_samples` images through the copy, compute a proxy loss (sum of per-tensor means from model output), backprop. For each Conv2d filter, importance = `(weight × gradient)²` summed across all spatial/in-channel dimensions. This approximates the diagonal Fisher information.

3. **Multi-objective metric** — for each conv layer, compute `s_i = Fisher_i / (norm_params × norm_MACs)`, where `norm_params` and `norm_MACs` are layer-wide totals normalized by the max across all layers. Layers with lower `s_i` are deemed less important.

4. **Filter selection** — pick the bottom `freeze_portion` layers by `s_i`, then within each pick the bottom `freeze_portion` filters by raw importance score.

5. **Freeze hooks** — register backward hooks on the *original* (non-copied) model's `Conv2d.weight` and `.bias` that zero out gradients for the selected filter indices.

## Integration

In `yolo_detector.py:113–125`, right before `model.train()`:

```python
tt = TinyTrain(self._model, tt_config)
tt.apply(data_yaml=data_yaml, device=..., imgsz=...)
results = self._model.train(**args)
tt.remove_hooks()
```

Hooks are removed after training so inference/evaluation is unaffected.

## Bugs Fixed

| Bug | Symptom | Fix |
|---|---|---|
| Device override ignored | Train ran on MPS despite `--device cuda:0` | Use `args.get("device")` instead of `tr.get("device")` |
| Dict model output | `unsupported ** or pow(): 'dict' and 'int'` | Handle dict/list/tensor outputs when computing proxy loss |
| Frozen model copy | `element 0 does not require grad` | Call `requires_grad_(True)` on the deep-copy before Fisher forward/backward |
| `requires_grad` guard on hooks | 0 hooks registered (model loads with `requires_grad=False`) | Remove the guard — hooks fire correctly once the training loop enables grads |
| Deep-copy of `Sequential` instead of `DetectionModel` | `Concat` forward crash (`torch.cat` got Tensor, not list) — YOLOv5 and YOLOv8 | Deep-copy `self.model` (`DetectionModel`) instead of `self.model.model` (`Sequential`) so skip-connection routing works |
