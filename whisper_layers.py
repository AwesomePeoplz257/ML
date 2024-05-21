from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, WhisperForConditionalGeneration
from transformers import WhisperForConditionalGeneration as ws
import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaleAndShift(nn.Module):
    def __init__(self, dims):
        super().__init__()

    def forward(self, x):
        return x * self.scale + self.shift


class LoraLinear(nn.Module):
    def __init__(self, linear, a=None, b=None, scale=None, shift=None, rank=4):
        super().__init__()
        self.weight = linear.weight
        self.bias = linear.bias if linear.bias is not None else None
        dim_in, dim_out = self.weight.shape
        self.a = a if a is not None else torch.randn(dim_in, rank)
        self.b = b if b is not None else torch.randn(rank, dim_out)
        self.scale = scale if scale is not None else nn.Parameter(torch.ones(1, 1, dim_out))
        self.shift = shift if shift is not None else nn.Parameter(torch.zeros(1, 1, dim_out))
        self.trainable = (self.a, self.b, self.scale, self.shift)

    def forward(self, x):
        weight = self.weight + self.a @ self.b
        return F.linear(x, weight, self.bias) * self.scale + self.shift

if __name__ == "__main__":
    print('f')
    print(*LoraLinear(nn.Linear(1,1)).parameters())