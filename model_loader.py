from whisper_layers import LoraLinear
from transformers import WhisperForConditionalGeneration as ws
import torch
import torch.nn as nn
from functools import reduce


def set_new_module(module, name, model):
    path = name.split('.')
    parent = reduce(getattr, [model] + path[:-1])
    setattr(parent, path[-1], module)


def lora_init(model, modules=None, unfreeze_last=False):
    trainable_params, _trainable_params_names = [] if modules is None else modules, []
    c = 0
    for name, module in model.named_modules():
        if (isinstance(module, nn.Linear) or isinstance(module, LoraLinear)) and not (
                unfreeze_last and name == "proj_out"):
            if modules is None:
                new_module = LoraLinear(module)
                trainable_params.append(new_module.trainable)
            else:
                a, b, scale, shift = modules[c]
                new_module = LoraLinear(module, a, b, scale, shift)
            set_new_module(new_module, name, model)
            c += 1
        elif isinstance(module, nn.LayerNorm):
            if modules is not None:
                weight, bias = modules[c]
                new_module = nn.LayerNorm(module.normalized_shape, module.eps, module.elementwise_affine)
                new_module.weight = weight
                new_module.bias = bias
                set_new_module(new_module, name, model)
            else:
                trainable_params.append((module.weight, module.bias))
            c += 1
    if modules is None and unfreeze_last:
        trainable_params.append((model.proj_out.weight,))
    # print(trainable_params)
    for param in model.parameters():
        param.requires_grad = False
    for module in trainable_params:
        for param in module:
            param.requires_grad = True
    print("Number of trainable parameters: ", sum(p.numel() for p in model.parameters() if p.requires_grad))
    return model, trainable_params


if __name__ == "__main__":
    model = ws.from_pretrained("openai/whisper-small")
    # print(model)
    model, params = lora_init(model)
    model, params = lora_init(model, modules=params)
    print(model)
    # print(model)
