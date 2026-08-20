import unittest
from collections import OrderedDict

from tools.moshi_state_dict import (
    expose_linear_weights_in_original_state_dict,
    is_original_format_state_dict,
    restore_linear_weights_from_exposed_state_dict,
)

ORIGINAL_KEYS = [
    "text_emb.weight",
    "transformer.layers.0.gating.linear_in.weight",
    "transformer.layers.0.gating.linear_out.weight",
    "transformer.layers.0.self_attn.in_proj_weight",
    # only the depformer exposes self_attn.out_proj, so this one must survive untouched
    "transformer.layers.0.self_attn.out_proj.weight",
    "transformer.layers.31.gating.linear_in.weight",
    "depformer.layers.0.gating.0.linear_in.weight",
    "depformer.layers.0.gating.7.linear_out.weight",
    "depformer.layers.5.self_attn.out_proj.weight",
    "out_norm.alpha",
]

EXPOSED_KEYS = [
    "text_emb.weight",
    "transformer.layers.0.gating.linear_in_weight",
    "transformer.layers.0.gating.linear_out_weight",
    "transformer.layers.0.self_attn.in_proj_weight",
    "transformer.layers.0.self_attn.out_proj.weight",
    "transformer.layers.31.gating.linear_in_weight",
    "depformer.layers.0.gating.0.linear_in_weight",
    "depformer.layers.0.gating.7.linear_out_weight",
    "depformer.layers.5.self_attn.out_proj_weight",
    "out_norm.alpha",
]


def _state_dict(keys: list[str]) -> OrderedDict:
    return OrderedDict((key, f"tensor::{key}") for key in keys)


class ExposeOriginalStateDictTests(unittest.TestCase):
    """`tools/clean_moshi.py` publishes original Moshi names.

    `MoshiForFinetuning.__init__` renames those parameters for DeepSpeed Zero-3, so an
    original-format checkpoint cannot be loaded without this remap.
    """

    def test_renames_every_exposed_parameter(self) -> None:
        result = expose_linear_weights_in_original_state_dict(_state_dict(ORIGINAL_KEYS))

        self.assertEqual(list(result.keys()), EXPOSED_KEYS)

    def test_keeps_the_temporal_transformer_out_proj_untouched(self) -> None:
        result = expose_linear_weights_in_original_state_dict(
            _state_dict(["transformer.layers.0.self_attn.out_proj.weight"])
        )

        self.assertEqual(list(result.keys()), ["transformer.layers.0.self_attn.out_proj.weight"])

    def test_preserves_values_and_order(self) -> None:
        result = expose_linear_weights_in_original_state_dict(_state_dict(ORIGINAL_KEYS))

        self.assertEqual(list(result.values()), [f"tensor::{key}" for key in ORIGINAL_KEYS])

    def test_is_the_inverse_of_the_restore_helper(self) -> None:
        exposed = _state_dict(EXPOSED_KEYS)

        round_tripped = expose_linear_weights_in_original_state_dict(
            restore_linear_weights_from_exposed_state_dict(exposed)
        )

        self.assertEqual(list(round_tripped.keys()), EXPOSED_KEYS)

    def test_leaves_an_already_exposed_state_dict_alone(self) -> None:
        result = expose_linear_weights_in_original_state_dict(_state_dict(EXPOSED_KEYS))

        self.assertEqual(list(result.keys()), EXPOSED_KEYS)


class OriginalFormatDetectionTests(unittest.TestCase):
    def test_detects_an_original_format_checkpoint(self) -> None:
        self.assertTrue(is_original_format_state_dict(ORIGINAL_KEYS))

    def test_rejects_an_exposed_checkpoint(self) -> None:
        self.assertFalse(is_original_format_state_dict(EXPOSED_KEYS))

    def test_does_not_treat_the_transformer_out_proj_as_original_format(self) -> None:
        self.assertFalse(
            is_original_format_state_dict(["transformer.layers.0.self_attn.out_proj.weight"])
        )


if __name__ == "__main__":
    unittest.main()
