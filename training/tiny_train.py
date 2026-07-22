"""
TinyTrain: Selective filter freezing for efficient fine-tuning.

Adapted from the TinyTrain framework for ultralytics YOLOv5/v8.
Estimates filter importance via Fisher information, then selectively
freezes the least important conv filters so only the critical ~10% train.
"""

import copy
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Dataset
from PIL import Image


def _get_conv_weight_names(model) -> list[str]:
    return [f"{name}.weight" for name, mod in model.named_modules()
            if isinstance(mod, nn.Conv2d)]


def _order_scores(imp_order):
    imp_order = np.array(imp_order, dtype=object)
    return imp_order[imp_order[:, 3].astype(float).argsort()]


def _order_and_ratios_yolov5(sorted_imp_order):
    avg_importance_layer = {}
    imp_order_layer = {}
    layer_names = np.unique(sorted_imp_order[:, 0])
    for layer_name in layer_names:
        layer_importances = sorted_imp_order[
            (sorted_imp_order[:, 0] == layer_name), 3
        ].astype(float)
        avg_importance = float(np.mean(layer_importances)) if layer_importances.size > 0 else 0
        avg_importance_layer[layer_name] = avg_importance
        filter_indices = sorted_imp_order[
            (sorted_imp_order[:, 0] == layer_name), 2
        ].astype(int)
        imp_order_layer[layer_name] = filter_indices
    sorted_layers = dict(sorted(avg_importance_layer.items(), key=lambda item: item[1]))
    return imp_order_layer, sorted_layers


def _compute_multi_obj_metric(ordered_layers, conv_param_dict, conv_mac_dict):
    max_param = max(conv_param_dict.values())
    max_mac = max(conv_mac_dict.values())
    normalized_params = {k: v / max_param for k, v in conv_param_dict.items()}
    normalized_macs = {k: v / max_mac for k, v in conv_mac_dict.items()}
    si_metric_dict = {}
    for layer_name, fisher_info in ordered_layers.items():
        matching_key = next((k for k in normalized_params if k in layer_name), None)
        if matching_key is not None:
            s_i = fisher_info / (normalized_params[matching_key] * normalized_macs[matching_key])
            si_metric_dict[layer_name] = s_i
    return dict(sorted(si_metric_dict.items(), key=lambda item: item[1]))


def _compute_conv_params_and_macs(model, input_size):
    conv_param_dict = {}
    conv_mac_dict = {}
    hooks = []
    device = next(model.parameters()).device

    def compute_mac_hook(module, input, output, name):
        if isinstance(module, nn.Conv2d):
            num_trainable_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
            conv_param_dict[name] = num_trainable_params
            output_dims = output.shape[2:]
            macs = (output_dims[0] * output_dims[1] * module.out_channels *
                    module.kernel_size[0] * module.kernel_size[1] *
                    (module.in_channels // module.groups))
            conv_mac_dict[name] = macs

    for name, layer in model.named_modules():
        if isinstance(layer, nn.Conv2d):
            hooks.append(
                layer.register_forward_hook(
                    lambda m, i, o, n=name: compute_mac_hook(m, i, o, n)
                )
            )

    model.eval()
    dummy_input = torch.randn(input_size).to(device)
    with torch.no_grad():
        model(dummy_input)

    for hook in hooks:
        hook.remove()
    return conv_param_dict, conv_mac_dict


def _get_layers_filters_to_freeze(sorted_si_metric, ordered_layerwise, freeze_portion):
    filters_to_freeze = {}
    num_layers_to_freeze = int(freeze_portion * len(sorted_si_metric))
    least_important_layers = list(sorted_si_metric.keys())[:num_layers_to_freeze]
    for layer_name in least_important_layers:
        if layer_name in ordered_layerwise:
            filters = ordered_layerwise[layer_name]
            num_freeze = int(freeze_portion * len(filters))
            filters_to_freeze[layer_name] = filters[:num_freeze]
    return filters_to_freeze


def _encode_layer_name(name):
    return name.replace(".", "_")


def _dict_to_nested_list(encoded_dict):
    nested_list = []
    for layer_name, filters in encoded_dict.items():
        filters_list = filters.tolist() if hasattr(filters, "tolist") else list(filters)
        nested_list.append([layer_name] + filters_list)
    return nested_list


class _SimpleImageDataset(Dataset):
    def __init__(self, img_dir: str, imgsz: int = 640):
        self.paths = list(Path(img_dir).glob("*.jpg")) + \
                     list(Path(img_dir).glob("*.png")) + \
                     list(Path(img_dir).glob("*.jpeg"))
        self.imgsz = imgsz

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        img = img.resize((self.imgsz, self.imgsz))
        arr = np.array(img, dtype=np.float32).transpose(2, 0, 1) / 255.0
        return torch.from_numpy(arr)


def _make_freeze_hook(frozen_indices):
    def hook(grad):
        grad_copy = grad.clone()
        grad_copy[frozen_indices] = 0
        return grad_copy
    return hook


class TinyTrain:
    """
    TinyTrain selective filter freezing for ultralytics YOLO models.

    Usage:
        tt = TinyTrain(yolo_model, config)
        tt.apply(data_yaml, device)
        # ... call model.train() as normal ...
        tt.remove_hooks()
    """

    def __init__(self, model, config: dict | None = None):
        self.model = model
        self.config = config or {}
        self.hooks = []

    def apply(self, data_yaml: str, device: str = "cpu", imgsz: int | list = 640):
        if isinstance(imgsz, (list, tuple)):
            imgsz = imgsz[0]
        print("  [TinyTrain] Estimating filter importance ...")
        detection_model = self.model
        model_copy = copy.deepcopy(detection_model)
        model_copy.train()
        for p in model_copy.parameters():
            p.requires_grad_(True)
        model_copy.to(device)

        data_yaml_path = Path(data_yaml)
        with open(data_yaml_path) as f:
            data_cfg = yaml.safe_load(f)
        train_path = data_cfg["train"]
        path_base = data_yaml_path.parent.resolve()
        if not Path(train_path).is_absolute():
            candidate = (path_base / train_path).resolve()
            if not candidate.exists() and train_path.startswith("../"):
                candidate = (path_base / train_path[3:]).resolve()
            train_path = str(candidate)

        batch_size = min(4, self.config.get("imp_batch_size", 4))
        num_stop = self.config.get("imp_samples", 1000)
        freeze_portion = self.config.get("freeze_portion", 0.9)

        dataset = _SimpleImageDataset(train_path, imgsz)
        if len(dataset) == 0:
            print("  [TinyTrain] WARNING: no training images found for importance estimation!")
            return
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

        base_params = [p for p in model_copy.parameters() if p.requires_grad]
        grad_accum = [torch.zeros_like(p) for p in base_params]
        num_processed = 0

        for batch in loader:
            if num_processed >= num_stop:
                break
            imgs = batch.to(device)
            model_copy.zero_grad()
            output = model_copy(imgs)
            if isinstance(output, dict):
                tensors = [v for v in output.values() if isinstance(v, torch.Tensor)]
                loss = sum(t.mean() for t in tensors) if tensors else torch.tensor(0.0, device=device)
            elif isinstance(output, (list, tuple)):
                loss = sum(o.mean() for o in output if isinstance(o, torch.Tensor))
            else:
                loss = torch.mean(output ** 2)
            loss.backward()
            bs = imgs.size(0)
            for i, p in enumerate(base_params):
                if p.grad is not None:
                    grad_accum[i] += p.grad.detach() * bs
            num_processed += bs

        num_processed = max(num_processed, 1)
        grad_avg = [g / num_processed for g in grad_accum]

        layer_names = [n for n, p in model_copy.named_parameters() if p.requires_grad]
        list_imp_yolo = []
        for i, p in enumerate(base_params):
            p_2d = p.detach().reshape(p.shape[0], -1)
            g_2d = grad_avg[i].reshape(grad_avg[i].shape[0], -1)
            imp = (p_2d * g_2d).sum(dim=1).pow(2).cpu().numpy()
            list_imp_yolo.append(imp)

        print(f"  [TinyTrain] Building multi-objective metric ...")
        conv_weight_names = _get_conv_weight_names(model_copy)
        conv_importance_weight = []
        for ind, name in enumerate(layer_names):
            if name not in conv_weight_names:
                continue
            scores = list_imp_yolo[ind]
            for filter_idx in range(scores.shape[0]):
                conv_importance_weight.append([name, ind, filter_idx, float(scores[filter_idx])])

        ordered_imp = _order_scores(np.array(conv_importance_weight))
        order_layerwise, order_layers = _order_and_ratios_yolov5(ordered_imp)
        conv_param_dict, conv_mac_dict = _compute_conv_params_and_macs(
            model_copy, (1, 3, imgsz, imgsz)
        )

        sorted_si = _compute_multi_obj_metric(order_layers, conv_param_dict, conv_mac_dict)
        layers_filters_to_freeze = _get_layers_filters_to_freeze(
            sorted_si, order_layerwise, freeze_portion
        )

        total_filters = sum(len(v) for v in layers_filters_to_freeze.values())
        print(f"  [TinyTrain] Freezing {total_filters} filters across "
              f"{len(layers_filters_to_freeze)} layers "
              f"({freeze_portion * 100:.0f}% freeze ratio)")

        self._apply_freeze_hooks(detection_model, layers_filters_to_freeze)
        del model_copy

    def _apply_freeze_hooks(self, internal_model, layers_filters_to_freeze):
        for name, module in internal_model.named_modules():
            weight_name = f"{name}.weight" if name else "weight"
            if weight_name in layers_filters_to_freeze:
                frozen_idx = set(layers_filters_to_freeze[weight_name])
                h = module.weight.register_hook(_make_freeze_hook(frozen_idx))
                self.hooks.append(h)

        for name, module in internal_model.named_modules():
            bias_name = f"{name}.bias" if name else "bias"
            weight_name = f"{name}.weight" if name else "weight"
            if weight_name in layers_filters_to_freeze and module.bias is not None:
                frozen_idx = set(layers_filters_to_freeze[weight_name])
                frozen_idx = {i for i in frozen_idx if i < module.bias.shape[0]}
                if frozen_idx:
                    h = module.bias.register_hook(_make_freeze_hook(frozen_idx))
                    self.hooks.append(h)

        print(f"  [TinyTrain] Registered {len(self.hooks)} gradient hooks")

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []
        print("  [TinyTrain] Removed gradient hooks")
