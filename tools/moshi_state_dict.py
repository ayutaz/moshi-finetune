"""Parameter-name translation between the original Moshi layout and the finetuning layout.

`MoshiForFinetuning.__init__` calls `expose_linear_weights_for_zero3`, which lifts a few
linear-layer weights up to their parent module so DeepSpeed Zero-3 can reach them. That
renames parameters, so a checkpoint written in the original Moshi layout — which is what
`tools/clean_moshi.py` publishes — does not load into `MoshiForFinetuning` as-is.

These helpers are pure string work on state-dict keys and import nothing heavy, so they
stay unit-testable without torch.

Exposed parameters:

- `transformer.layers[*].gating.linear_in_weight`
- `transformer.layers[*].gating.linear_out_weight`
- `depformer.layers[*].gating[*].linear_in_weight`
- `depformer.layers[*].gating[*].linear_out_weight`
- `depformer.layers[*].self_attn.out_proj_weight`

Only the depformer exposes `self_attn.out_proj`; the temporal transformer keeps its own
`self_attn.out_proj.weight` in both layouts.
"""

from __future__ import annotations

import re
from collections import OrderedDict
from collections.abc import Iterable

_EXPOSED_TO_ORIGINAL = (
    (r"^transformer\.layers\.\d+\.gating\.linear_in_weight$", "linear_in_weight", "linear_in.weight"),
    (r"^transformer\.layers\.\d+\.gating\.linear_out_weight$", "linear_out_weight", "linear_out.weight"),
    (r"^depformer\.layers\.\d+\.gating\.\d+\.linear_in_weight$", "linear_in_weight", "linear_in.weight"),
    (r"^depformer\.layers\.\d+\.gating\.\d+\.linear_out_weight$", "linear_out_weight", "linear_out.weight"),
    (r"^depformer\.layers\.\d+\.self_attn\.out_proj_weight$", "out_proj_weight", "out_proj.weight"),
)

_ORIGINAL_TO_EXPOSED = (
    (r"^transformer\.layers\.\d+\.gating\.linear_in\.weight$", "linear_in.weight", "linear_in_weight"),
    (r"^transformer\.layers\.\d+\.gating\.linear_out\.weight$", "linear_out.weight", "linear_out_weight"),
    (r"^depformer\.layers\.\d+\.gating\.\d+\.linear_in\.weight$", "linear_in.weight", "linear_in_weight"),
    (r"^depformer\.layers\.\d+\.gating\.\d+\.linear_out\.weight$", "linear_out.weight", "linear_out_weight"),
    (r"^depformer\.layers\.\d+\.self_attn\.out_proj\.weight$", "out_proj.weight", "out_proj_weight"),
)


def _compile(rules: tuple) -> tuple:
    return tuple((re.compile(pattern), source, target) for pattern, source, target in rules)


_EXPOSED_TO_ORIGINAL_RULES = _compile(_EXPOSED_TO_ORIGINAL)
_ORIGINAL_TO_EXPOSED_RULES = _compile(_ORIGINAL_TO_EXPOSED)


def _rename_key(key: str, rules: tuple) -> str:
    for pattern, source, target in rules:
        if pattern.match(key):
            return key.replace(source, target)
    return key


def _rename(state_dict: OrderedDict, rules: tuple) -> OrderedDict:
    return OrderedDict((_rename_key(key, rules), value) for key, value in state_dict.items())


def restore_linear_weights_from_exposed_state_dict(state_dict: OrderedDict) -> OrderedDict:
    """Convert finetuning parameter names back to the original Moshi names."""
    return _rename(state_dict, _EXPOSED_TO_ORIGINAL_RULES)


def expose_linear_weights_in_original_state_dict(state_dict: OrderedDict) -> OrderedDict:
    """Convert original Moshi parameter names to the exposed finetuning names."""
    return _rename(state_dict, _ORIGINAL_TO_EXPOSED_RULES)


def is_original_format_state_dict(keys: Iterable[str]) -> bool:
    """Report whether a checkpoint uses the original Moshi parameter names."""
    return any(
        pattern.match(key) for key in keys for pattern, _, _ in _ORIGINAL_TO_EXPOSED_RULES
    )


def count_original_format_keys(keys: Iterable[str]) -> int:
    """Count how many parameters would be renamed when loading an original-format checkpoint."""
    return sum(
        1
        for key in keys
        if any(pattern.match(key) for pattern, _, _ in _ORIGINAL_TO_EXPOSED_RULES)
    )
