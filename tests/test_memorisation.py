import unittest

from tools.experiment_data import _near_duplicate
from tools.memorisation import (
    containment,
    memorisation_verdict,
    reproduced_training_texts,
)

# One of the 80 corpus train sentences M3 fine-tunes on.
TRAINING = "ニューイングランド風は、牛乳をベースとした、白いクリームスープであり、ボストンクラムチャウダーとも呼ばれる。"

# A generation that reproduces it verbatim inside ordinary conversational filler. This is
# what memorisation actually looks like in a dialogue model: the training sentence comes
# back whole, wrapped in text the model made up.
GENERATION_WITH_TRAINING_TEXT = (
    "はい、承知しました。ご質問ありがとうございます。少しお時間をいただきますね。"
    + TRAINING
    + "以上がご説明になります。他にお聞きになりたいことがあれば、どうぞおっしゃってください。"
)


class WhyJaccardIsBlindTests(unittest.TestCase):
    """The instrument already in the repository cannot see this, which is why a new one exists."""

    def test_the_existing_near_duplicate_check_misses_an_embedded_training_sentence(self) -> None:
        self.assertFalse(_near_duplicate(TRAINING, GENERATION_WITH_TRAINING_TEXT))

    def test_containment_sees_it_exactly(self) -> None:
        # Symmetric overlap is diluted by everything the generation added; asking what
        # fraction of the TRAINING sentence survives is not.
        self.assertEqual(containment(TRAINING, GENERATION_WITH_TRAINING_TEXT), 1.0)


class ContainmentTests(unittest.TestCase):
    def test_an_unrelated_pair_scores_low(self) -> None:
        self.assertLess(
            containment("こんにちは、今日はいい天気ですね", "量子力学の基礎方程式について"), 0.2
        )

    def test_it_is_asymmetric_by_design(self) -> None:
        short, long = (
            "白いクリームスープ",
            "白いクリームスープであり、ボストンクラムチャウダーとも呼ばれる",
        )
        self.assertEqual(containment(short, long), 1.0)
        self.assertLess(containment(long, short), 1.0)

    def test_empty_text_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            containment("", "なにか")


class ReproducedTrainingTextTests(unittest.TestCase):
    def test_it_names_which_training_sentence_came_back(self) -> None:
        hits = reproduced_training_texts(
            GENERATION_WITH_TRAINING_TEXT, [TRAINING, "まったく無関係な文章です"], threshold=0.8
        )
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["training_text"], TRAINING)
        self.assertTrue(hits[0]["exact_substring"])

    def test_an_original_generation_produces_no_hits(self) -> None:
        hits = reproduced_training_texts(
            "今日はとても良いお天気ですので、少し散歩でもいたしましょうか。",
            [TRAINING],
            threshold=0.8,
        )
        self.assertEqual(hits, [])

    def test_a_paraphrase_below_the_threshold_is_not_a_hit(self) -> None:
        hits = reproduced_training_texts("白いスープの話をしました。", [TRAINING], threshold=0.8)
        self.assertEqual(hits, [])


class MemorisationVerdictTests(unittest.TestCase):
    def test_verbatim_reproduction_alone_is_memorisation(self) -> None:
        verdict = memorisation_verdict(
            seen_delta=0.05, heldout_delta=0.04, verbatim_hits=1, min_delta=0.02
        )
        self.assertEqual(verdict["verdict"], "memorisation")
        self.assertIn("verbatim", verdict["reason"])

    def test_seen_improving_while_heldout_does_not_is_memorisation(self) -> None:
        # RUBRIC.md: seenだけが改善してheld-outが改善しないcheckpointは暗記と判定する
        verdict = memorisation_verdict(
            seen_delta=0.09, heldout_delta=0.001, verbatim_hits=0, min_delta=0.02
        )
        self.assertEqual(verdict["verdict"], "memorisation")

    def test_both_improving_is_generalisation(self) -> None:
        verdict = memorisation_verdict(
            seen_delta=0.05, heldout_delta=0.04, verbatim_hits=0, min_delta=0.02
        )
        self.assertEqual(verdict["verdict"], "generalisation")

    def test_neither_improving_is_not_memorisation(self) -> None:
        # It failed, but it did not memorise; calling it memorisation would misattribute.
        verdict = memorisation_verdict(
            seen_delta=0.001, heldout_delta=0.0, verbatim_hits=0, min_delta=0.02
        )
        self.assertEqual(verdict["verdict"], "no-improvement")

    def test_heldout_improving_without_seen_is_reported_as_odd(self) -> None:
        verdict = memorisation_verdict(
            seen_delta=0.0, heldout_delta=0.05, verbatim_hits=0, min_delta=0.02
        )
        self.assertEqual(verdict["verdict"], "inconsistent")


if __name__ == "__main__":
    unittest.main()
