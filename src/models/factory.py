# One function that builds a model from model_name in the YAML config

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch.nn as nn
from torchvision.models import DenseNet121_Weights, densenet121


@dataclass(frozen=True)
class ModelParts:
    model: nn.Module
    backbone_params: list[nn.Parameter]
    head_params: list[nn.Parameter]

BuilderFn = Callable[[dict[str, Any], int], ModelParts]

def _build_densenet121(cfg: dict[str, Any], num_classes: int) -> ModelParts:
    pretrained = bool(cfg.get("pretrained", True))
    weights = DenseNet121_Weights.DEFAULT if pretrained else None

    model = densenet121(weights=weights)

    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, num_classes)

    backbone_params = list(model.features.parameters())
    head_params = list(model.classifier.parameters())

    return ModelParts(
        model=model,
        backbone_params=backbone_params,
        head_params=head_params,
    )


MODEL_REGISTRY: dict[str, BuilderFn] = {
    "densenet121": _build_densenet121,
}


def build_model(cfg: dict[str, Any], num_classes: int) -> nn.Module:
    model_name = str(cfg["model_name"]).lower()

    try:
        builder = MODEL_REGISTRY[model_name]
    except KeyError as exc:
        available = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(
            f"Unsupported model_name '{model_name}'. "
            f"Available models: {available}"
        ) from exc
    
    parts = builder(cfg, num_classes=num_classes)
    return parts.model


def build_parameter_groups(
        model: nn.Module,
        cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    model_name = str(cfg["model_name"]).lower()
    optimizer_cfg = cfg["optimizer"]

    if model_name == "densenet121":
        head_params = list(model.classifier.parameters())
        head_param_ids = {id(param) for param in head_params}

        backbone_params = [
            param
            for param in model.parameters()
            if id(param) not in head_param_ids
        ]
    else:
        raise ValueError(f"No parameter-group rule defined for '{model_name}'.")
    
    return [
        {
            "params": backbone_params,
            "lr": float(optimizer_cfg["backbone_lr"]),
            "weight_decay": float(optimizer_cfg["weight_decay"]),
        },
        {
            "params": head_params,
            "lr": float(optimizer_cfg["head_lr"]),
            "weight_decay": float(optimizer_cfg["weight_decay"]),
        }
    ]