import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from einops import rearrange
from moshi.models import LMModel
from moshi.modules.gating import (
    gating_forward_kernel,
)
from moshi.modules.transformer import (
    StreamingTransformer,
    create_sin_embedding,
    multi_linear,
)
from safetensors import safe_open
from safetensors.torch import load_file, load_model, save_file

from tools.moshi_state_dict import (
    count_original_format_keys,
    expose_linear_weights_in_original_state_dict,
    is_original_format_state_dict,
    restore_linear_weights_from_exposed_state_dict,
)


def expose_linear_weights_for_zero3(
    moshi_lm: LMModel,
) -> None:
    """Exposes linear layer weights at their parent modules for DeepSpeed Zero-3 compatibility.

    In DeepSpeed Zero-3, weights of child modules cannot be accessed directly, so we need to
    restructure the model by exposing weights from linear layers at their
    parent modules while removing the original linear modules to maintain compatibility.

    Target modules:
    - `transformer.layers[*].gating.linear_in`
    - `transformer.layers[*].gating.linear_out`
    - `depformer.layers[*].gating[*].linear_in`
    - `depformer.layers[*].gating[*].linear_out`
    - `depformer.layers[*].self_attn.out_proj`
    """
    if isinstance(moshi_lm.transformer, StreamingTransformer):
        for layer in moshi_lm.transformer.layers:
            layer.gating.linear_in_weight = layer.gating.linear_in.weight
            layer.gating.linear_out_weight = layer.gating.linear_out.weight
            del layer.gating.linear_in, layer.gating.linear_out
    if isinstance(moshi_lm.depformer, StreamingTransformer):
        for layer in moshi_lm.depformer.layers:
            for gating in layer.gating:
                gating.linear_in_weight = gating.linear_in.weight
                gating.linear_out_weight = gating.linear_out.weight
                del gating.linear_in, gating.linear_out
            layer.self_attn.out_proj_weight = layer.self_attn.out_proj.weight
            del layer.self_attn.out_proj


def activation_gating_forward(self, x: torch.Tensor):
    """
    Most of the code in this function is copied from `moshi.modules.gating.ActivationGating.forward`
    with the exception of accessing own linear weights.
    """
    return gating_forward_kernel(
        # use the exposed linear weights
        self.linear_in_weight,
        self.linear_out_weight,
        self.activation,
        x,
    )


def mha_forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor):
    """
    Most of the code in this function is copied from `moshi.modules.transformer.StreamingMultiheadAttention.forward`
    with the exception of accessing own linear weights.
    """
    state = self._streaming_state
    T = query.shape[1]

    if state is None:
        offset = torch.zeros(1, device=query.device, dtype=torch.long)
        offset_cpu = 0
    else:
        assert self.causal, "Streaming only available for causal"
        offset = state.offset
        offset_cpu = state.offset_cpu

    if self.weights_per_step:
        projected = multi_linear(self.weights_per_step, self.in_proj_weight, query, offset_cpu)
    else:
        projected = nn.functional.linear(query, self.in_proj_weight)
    q, k, v = rearrange(projected, "b t (p h d) -> p b h t d", p=3, h=self.num_heads)

    if self.rope:
        q, k = self.rope(q, k, offset, time_before_heads=False)

    k, v, pos_k = self._complete_kv(k, v)
    if self.causal:
        pos_k = pos_k.view(1, -1)
        pos_q = offset + torch.arange(T, device=q.device, dtype=torch.long).view(-1, 1)
        delta = pos_q - pos_k
        attn_bias = (pos_k >= 0) & (delta >= 0)
        if self.context is not None:
            attn_bias = attn_bias & (delta < self.context)
    else:
        attn_bias = None
    x = F.scaled_dot_product_attention(q, k, v, attn_bias, dropout_p=0.0)

    x = rearrange(x, "b h t d -> b t (h d)")
    if self.weights_per_step:
        # Use the exposed linear weights (out_proj_weight)
        x = multi_linear(self.weights_per_step, self.out_proj_weight, x, offset_cpu)
    else:
        x = self.out_proj(x)
    if state is not None:
        state.offset.add_(T)
        state.offset_cpu += T
    return x


def transformer_forward(self, x: torch.Tensor, *args, **kwargs):
    """
    Most of the code in this function is copied from `moshi.modules.transformer.StreamingTransformer.forward`
    with the exception of activation checkpointing.
    """
    B, T, C = x.shape

    state = self._streaming_state
    if state is None:
        offset = torch.zeros(1, dtype=torch.long, device=x.device)
    else:
        offset = state.offset

    if self.positional_embedding in {"sin", "sin_rope"}:
        positions = torch.arange(T, device=x.device).view(1, -1, 1)
        positions = positions + offset.view(-1, 1, 1)
        pos_emb = create_sin_embedding(positions, C, max_period=self.max_period, dtype=x.dtype)
        x = x + self.positional_scale * pos_emb

    for layer in self.layers:
        if self.activation_checkpointing:
            x = self.checkpointing_func(layer, x)
        else:
            x = layer(x)  # , *args, **kwargs)

    if state is not None:
        state.offset.add_(T)
    return x


class MoshiForFinetuning(LMModel):
    """
    Moshi language model for finetuning.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # DeepSpeed Zero-3 compatibility
        ## 1. Expose linear layer weights
        expose_linear_weights_for_zero3(self)
        ## 2. Apply patches for forward functions
        for layer in self.transformer.layers:
            layer.gating.forward = activation_gating_forward.__get__(layer.gating)
        for layer in self.depformer.layers:
            for gating in layer.gating:
                gating.forward = activation_gating_forward.__get__(gating)
        for layer in self.depformer.layers:
            layer.self_attn.forward = mha_forward.__get__(layer.self_attn)

        # Implement activation checkpointing
        self.transformer.activation_checkpointing = False
        self.transformer.forward = transformer_forward.__get__(self.transformer)
        self.depformer.activation_checkpointing = False
        self.depformer.forward = transformer_forward.__get__(self.depformer)

    def enable_activation_checkpointing(self, checkpointing_func):
        self.transformer.activation_checkpointing = True
        self.transformer.checkpointing_func = checkpointing_func
        self.depformer.activation_checkpointing = True
        self.depformer.checkpointing_func = checkpointing_func

    def disable_activation_checkpointing(self):
        self.transformer.activation_checkpointing = False
        self.depformer.activation_checkpointing = False

    @classmethod
    def from_original_moshi_lm(
        cls,
        moshi_lm: LMModel,
        moshi_lm_kwargs: dict,
    ) -> "MoshiForFinetuning":
        """
        Initialize `MoshiForFinetuning` from the original `LMModel`.
        """
        # Expose linear layer weights for DeepSpeed Zero-3 compatibility
        expose_linear_weights_for_zero3(moshi_lm)
        state_dict = moshi_lm.state_dict()
        device = next(moshi_lm.parameters()).device
        dtype = next(moshi_lm.parameters()).dtype

        # Clear the original model to save memory
        del moshi_lm

        # Initialize the new model
        moshi_lm = cls(device=device, dtype=dtype, **moshi_lm_kwargs).to(device=device, dtype=dtype)
        moshi_lm.load_state_dict(state_dict, strict=True)

        # Store the kwargs for the later use
        moshi_lm.moshi_lm_kwargs = moshi_lm_kwargs

        return moshi_lm

    def to_original_moshi_lm(self) -> LMModel:
        """
        Convert the model to the original `LMModel`.
        """
        state_dict = self.state_dict()
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype

        # Convert the state dict to the original format
        state_dict = restore_linear_weights_from_exposed_state_dict(state_dict)

        # Initialize the original model
        moshi_lm = LMModel(device=device, dtype=dtype, **self.moshi_lm_kwargs).to(
            device=device, dtype=dtype
        )
        moshi_lm.load_state_dict(state_dict, strict=True)

        return moshi_lm

    def save_pretrained(self, save_dir: str):
        """
        Save the model to the given directory.
        """
        os.makedirs(save_dir, exist_ok=True)
        # Save the model
        save_file(self.state_dict(), os.path.join(save_dir, "model.safetensors"))
        # Save the kwargs
        with open(os.path.join(save_dir, "moshi_lm_kwargs.json"), "w") as f:
            json.dump(self.moshi_lm_kwargs, f, indent=4)

    @classmethod
    def from_pretrained(
        cls,
        save_dir: str,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.bfloat16,
    ) -> "MoshiForFinetuning":
        """
        Load the model from the given directory.
        """
        # Load the kwargs
        with open(os.path.join(save_dir, "moshi_lm_kwargs.json")) as f:
            moshi_lm_kwargs = json.load(f)
        # Initialize the model
        moshi_lm = cls(device=device, dtype=dtype, **moshi_lm_kwargs).to(device=device, dtype=dtype)
        # Load the model
        weights_path = os.path.join(save_dir, "model.safetensors")
        with safe_open(weights_path, framework="pt") as weights:
            checkpoint_keys = list(weights.keys())
        if is_original_format_state_dict(checkpoint_keys):
            # Checkpoints published through `tools/clean_moshi.py` keep the original Moshi
            # names, which do not match the ones `expose_linear_weights_for_zero3` creates.
            print(
                f"Loading an original-format checkpoint; renaming "
                f"{count_original_format_keys(checkpoint_keys)} exposed parameters"
            )
            state_dict = load_file(weights_path, device=str(device))
            moshi_lm.load_state_dict(
                expose_linear_weights_in_original_state_dict(state_dict), strict=True
            )
        else:
            load_model(moshi_lm, weights_path)

        moshi_lm.moshi_lm_kwargs = moshi_lm_kwargs

        return moshi_lm
