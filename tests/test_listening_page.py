import unittest

from tools.build_listening_page import blind_order, tally_judgements


class BlindOrderTests(unittest.TestCase):
    """The plan asks for a blind A/B, so which system plays first must not be guessable
    from the page and must not always be the same one.
    """

    def test_is_deterministic_for_the_same_seed(self) -> None:
        first = blind_order("tts-01", seed=7)
        second = blind_order("tts-01", seed=7)

        self.assertEqual(first, second)

    def test_returns_both_systems_once(self) -> None:
        order = blind_order("tts-01", seed=7)

        self.assertEqual(sorted(order), ["A_base", "B_adapted"])

    def test_a_different_seed_can_flip_the_order(self) -> None:
        orders = {blind_order("tts-01", seed=s) for s in range(20)}

        self.assertEqual(len(orders), 2)

    def test_spreads_both_orders_across_a_sentence_set(self) -> None:
        ids = [f"tts-{index:02d}" for index in range(1, 31)]

        firsts = [blind_order(i, seed=20260820)[0] for i in ids]
        adapted_first = firsts.count("B_adapted")

        self.assertGreater(adapted_first, 5)
        self.assertLess(adapted_first, 25)


class TallyTests(unittest.TestCase):
    def test_counts_votes_per_system(self) -> None:
        judgements = {"tts-01": "B_adapted", "tts-02": "B_adapted", "tts-03": "A_base"}

        tally = tally_judgements(judgements)

        self.assertEqual(tally["B_adapted"], 2)
        self.assertEqual(tally["A_base"], 1)
        self.assertEqual(tally["total"], 3)

    def test_counts_ties_separately(self) -> None:
        tally = tally_judgements({"tts-01": "tie", "tts-02": "B_adapted"})

        self.assertEqual(tally["tie"], 1)
        self.assertEqual(tally["B_adapted"], 1)

    def test_an_empty_set_is_all_zero(self) -> None:
        tally = tally_judgements({})

        self.assertEqual(tally["total"], 0)
        self.assertEqual(tally["A_base"], 0)
