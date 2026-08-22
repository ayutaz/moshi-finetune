import unittest

from tools.experiment_data import apply_row_overrides, resolve_splits


class ResolveSplitsTests(unittest.TestCase):
    def test_without_an_override_the_seed_decides(self) -> None:
        hashed = resolve_splits(["a", "b", "c"], seed="s", override=None)
        self.assertEqual(set(hashed), {"a", "b", "c"})

    def test_an_override_replaces_the_hash_entirely(self) -> None:
        # M3's split is committed in split-map-v1.json and shared by both datasets. Letting
        # the hash re-derive it would give the two datasets different splits from the same
        # scripts, and the paired comparison would no longer be paired.
        override = {"a": "train", "b": "dev", "c": "train"}
        self.assertEqual(resolve_splits(["a", "b", "c"], seed="s", override=override), override)

    def test_a_group_missing_from_the_override_is_an_error(self) -> None:
        # Silently falling back to the hash for the missing one is how half a dataset ends
        # up on a different split than the other half.
        with self.assertRaises(ValueError) as caught:
            resolve_splits(["a", "b"], seed="s", override={"a": "train"})
        self.assertIn("b", str(caught.exception))

    def test_an_override_naming_an_unknown_group_is_an_error(self) -> None:
        with self.assertRaises(ValueError) as caught:
            resolve_splits(["a"], seed="s", override={"a": "train", "ghost": "dev"})
        self.assertIn("ghost", str(caught.exception))

    def test_an_unknown_split_name_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            resolve_splits(["a"], seed="s", override={"a": "holdout"})


class ApplyRowOverridesTests(unittest.TestCase):
    def test_per_row_derivation_replaces_the_dataset_wide_one(self) -> None:
        # Every V dialogue quotes a different corpus sentence, so one derivation for the
        # whole dataset records nothing usable - and the leakage assertion checks per row.
        rows = [
            {"group_id": "v-001", "derivation": []},
            {"group_id": "v-002", "derivation": []},
        ]
        overrides = {
            "v-001": {"derivation": ["tsukuyomi-corpus-v1:VOICEACTRESS100_001"]},
            "v-002": {"derivation": ["tsukuyomi-corpus-v1:VOICEACTRESS100_002"]},
        }
        applied = apply_row_overrides(rows, overrides)
        self.assertEqual(applied[0]["derivation"], ["tsukuyomi-corpus-v1:VOICEACTRESS100_001"])
        self.assertEqual(applied[1]["derivation"], ["tsukuyomi-corpus-v1:VOICEACTRESS100_002"])

    def test_fields_not_overridden_are_left_alone(self) -> None:
        rows = [{"group_id": "v-001", "text": "kept", "derivation": []}]
        applied = apply_row_overrides(rows, {"v-001": {"derivation": ["x"]}})
        self.assertEqual(applied[0]["text"], "kept")

    def test_a_row_with_no_override_is_an_error(self) -> None:
        with self.assertRaises(ValueError):
            apply_row_overrides([{"group_id": "v-001"}], {})

    def test_an_override_for_an_absent_row_is_an_error(self) -> None:
        with self.assertRaises(ValueError) as caught:
            apply_row_overrides([{"group_id": "v-001"}], {"v-001": {}, "v-999": {}})
        self.assertIn("v-999", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
