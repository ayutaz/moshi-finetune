import hashlib
import json
import unittest
from pathlib import Path

from tools.dialogue_scripts import (
    BACKCHANNEL_POOL,
    MIN_FRAGMENT_CHARS,
    TimelineSpec,
    assign_backchannels,
    backchannel_clashes,
    build_v2_scripts,
    collect_eval_texts,
    leading_clause,
    project_dialogue_seconds,
    project_frames,
    project_script_frames,
    project_seconds,
    split_at_central_comma,
    split_dialogue_row,
    summarise_split_points,
    summarise_structure,
    validate_scripts,
)

SPEC = TimelineSpec(lead_in_seconds=0.5, gap_seconds=0.4, frame_rate_hz=12.5)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = REPOSITORY_ROOT / "experiments" / "tsukuyomi_ojousama"


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


class ProjectionTests(unittest.TestCase):
    def test_seconds_are_mora_over_rate(self) -> None:
        self.assertAlmostEqual(project_seconds(60, mora_per_second=6.0), 10.0)

    def test_a_faster_rate_shortens_the_projection(self) -> None:
        # The floor gate must assume the FAST end: if the voice speaks faster than
        # projected, the dialogue lands shorter than planned and can slip under the floor.
        fast = project_seconds(60, mora_per_second=8.0)
        slow = project_seconds(60, mora_per_second=5.0)
        self.assertLess(fast, slow)

    def test_a_non_positive_rate_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            project_seconds(60, mora_per_second=0.0)

    def test_frames_round_down_so_the_gate_never_overstates(self) -> None:
        self.assertEqual(project_frames(1.99, frame_rate_hz=12.5), 24)

    def test_a_three_turn_dialogue_sums_turns_gaps_and_lead_in(self) -> None:
        # 0.5 lead-in + 2.0 + 0.4 + 3.0 + 0.4 + 2.0
        total = project_dialogue_seconds([2.0, 3.0, 2.0], spec=SPEC)
        self.assertAlmostEqual(total, 8.3)

    def test_a_single_turn_dialogue_has_no_gaps(self) -> None:
        self.assertAlmostEqual(project_dialogue_seconds([3.0], spec=SPEC), 3.5)

    def test_an_empty_dialogue_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            project_dialogue_seconds([], spec=SPEC)

    def test_script_frames_use_an_injected_mora_counter(self) -> None:
        # pyopenjtalk is not installed in the test environment, so the projection has to be
        # checkable without it.
        frames = project_script_frames(
            ["あ", "い", "う"], mora_per_second=7.5, spec=SPEC, mora_of=lambda text: 15
        )
        # 0.5 + 3 x 2.0 seconds + 2 x 0.4 = 7.3 s -> 91 frames
        self.assertEqual(frames, 91)

    def test_an_extra_turn_lengthens_the_projection_by_a_turn_and_a_gap(self) -> None:
        # Adding the backchannel adds its own duration plus one gap, and nothing else.
        three = project_script_frames(["a", "b", "c"], mora_per_second=10.0, spec=SPEC, mora_of=len)
        five = project_script_frames(
            ["a", "b", "x", "c", "d"], mora_per_second=10.0, spec=SPEC, mora_of=len
        )
        self.assertGreater(five, three)


class SplitAtCentralCommaTests(unittest.TestCase):
    SENTENCE = "また、東寺のように、五大明王と呼ばれる、主要な明王の中央に配されることも多い。"

    def test_it_cuts_at_the_comma_nearest_the_middle(self) -> None:
        left, right = split_at_central_comma(self.SENTENCE)
        self.assertEqual(left, "また、東寺のように、五大明王と呼ばれる、")
        self.assertEqual(right, "主要な明王の中央に配されることも多い。")

    def test_the_fragments_rebuild_the_sentence_exactly(self) -> None:
        # V-real's audio for A's turn IS the corpus recording. A fragment that is not a
        # substring of the sentence has no audio to go with it.
        left, right = split_at_central_comma(self.SENTENCE)
        self.assertEqual(left + right, self.SENTENCE)

    def test_the_comma_stays_on_the_left_fragment(self) -> None:
        left, _ = split_at_central_comma(self.SENTENCE)
        self.assertTrue(left.endswith("、"))

    def test_a_sentence_without_a_comma_does_not_split(self) -> None:
        self.assertIsNone(split_at_central_comma("クィーンズアベニューアルファに所属している。"))

    def test_a_comma_that_would_leave_a_stub_does_not_split(self) -> None:
        # The only comma sits two characters in; a two-character turn is a hiccup, not a turn.
        self.assertIsNone(split_at_central_comma("ああ、いいいいいいいいいい。"))

    def test_the_floor_applies_to_the_right_fragment_too(self) -> None:
        self.assertIsNone(split_at_central_comma("いいいいいいいいいい、ああ。"))

    def test_a_lower_floor_admits_a_split_the_default_refuses(self) -> None:
        self.assertEqual(
            split_at_central_comma("ああ、いいいいいいいいいい。", min_fragment_chars=3),
            ("ああ、", "いいいいいいいいいい。"),
        )

    def test_ties_go_to_the_earlier_comma(self) -> None:
        # Both commas sit the same distance from the centre; the result must not depend on
        # iteration order.
        left, right = split_at_central_comma("ああああ、ああ、ああああ", min_fragment_chars=2)
        self.assertEqual((left, right), ("ああああ、", "ああ、ああああ"))

    def test_a_non_positive_floor_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            split_at_central_comma(self.SENTENCE, min_fragment_chars=0)

    def test_a_multi_character_comma_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            split_at_central_comma(self.SENTENCE, comma="、、")


class LeadingClauseTests(unittest.TestCase):
    def test_it_stops_at_the_first_comma(self) -> None:
        self.assertEqual(leading_clause("なるほど、中央に配されるのですね。"), "なるほど")

    def test_it_stops_at_a_full_stop_when_there_is_no_comma(self) -> None:
        self.assertEqual(leading_clause("なるほど。"), "なるほど")

    def test_a_clause_with_no_punctuation_is_the_whole_text(self) -> None:
        self.assertEqual(leading_clause("なるほど"), "なるほど")


class AssignBackchannelsTests(unittest.TestCase):
    POOL = ("ええ。", "はい。", "なるほど。")

    def _followers(self, count: int) -> list[str]:
        return [f"それは{index}のことですね。" for index in range(count)]

    def test_every_seam_gets_one_backchannel(self) -> None:
        chosen = assign_backchannels(self._followers(7), pool=self.POOL, seed=1)
        self.assertEqual(len(chosen), 7)
        self.assertTrue(set(chosen) <= set(self.POOL))

    def test_the_pool_is_spread_evenly(self) -> None:
        # One phrase in every seam would be exactly the kind of always-true pattern this
        # rebuild exists to remove, so the counts have to stay within one of each other.
        chosen = assign_backchannels(self._followers(7), pool=self.POOL, seed=1)
        counts = [chosen.count(phrase) for phrase in self.POOL]
        self.assertEqual(sorted(counts), [2, 2, 3])

    def test_the_same_seed_gives_the_same_order(self) -> None:
        followers = self._followers(9)
        first = assign_backchannels(followers, pool=self.POOL, seed=20260825)
        second = assign_backchannels(followers, pool=self.POOL, seed=20260825)
        self.assertEqual(first, second)

    def test_the_seed_actually_decides_the_order(self) -> None:
        followers = self._followers(12)
        orders = {
            tuple(assign_backchannels(followers, pool=self.POOL, seed=seed)) for seed in range(1, 6)
        }
        self.assertGreater(len(orders), 1)

    def test_a_backchannel_is_kept_away_from_a_follower_that_opens_the_same_way(self) -> None:
        followers = ["なるほど、中央に配されるのですね。", "それは初耳です。", "よくわかりました。"]
        chosen = assign_backchannels(followers, pool=self.POOL, seed=3)
        self.assertNotEqual(leading_clause(chosen[0]), "なるほど")
        self.assertEqual(backchannel_clashes(chosen, followers), [])

    def test_the_swap_does_not_change_the_counts(self) -> None:
        followers = ["なるほど、そうですか。"] + self._followers(5)
        chosen = assign_backchannels(followers, pool=self.POOL, seed=3)
        counts = [chosen.count(phrase) for phrase in self.POOL]
        self.assertEqual(sorted(counts), [2, 2, 2])

    def test_an_empty_pool_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            assign_backchannels(self._followers(3), pool=(), seed=1)

    def test_a_pool_that_repeats_a_phrase_is_rejected(self) -> None:
        # A repeated phrase would quietly double its share of the seams.
        with self.assertRaises(ValueError):
            assign_backchannels(self._followers(3), pool=("ええ。", "ええ。"), seed=1)

    def test_clashes_that_could_not_be_swapped_away_are_reported(self) -> None:
        # Every follower opens with the only phrase in the pool, so no swap can help; the
        # count has to be reported rather than silently accepted.
        followers = ["ええ、そうですね。", "ええ、たしかに。"]
        chosen = assign_backchannels(followers, pool=("ええ。",), seed=1)
        self.assertEqual(backchannel_clashes(chosen, followers), [0, 1])

    def test_mismatched_lengths_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            backchannel_clashes(["ええ。"], ["a", "b"])


class SplitDialogueRowTests(unittest.TestCase):
    SENTENCE = "コンピュータゲームのメーカーや、業界団体などに関連する人物のカテゴリ。"

    def _row(self) -> dict:
        return {
            "schema_version": 1,
            "dialogue_id": "v-003",
            "source_artifact_id": "tsukuyomi-corpus-v1:VOICEACTRESS100_003",
            "split": "train",
            "turns": [
                {"speaker": "B", "text": "この分類は、どういう人たちをまとめたものでしょうか。"},
                {"speaker": "A", "text": self.SENTENCE},
                {"speaker": "B", "text": "かなり広い括りなのですね。"},
            ],
            "projected_frames_fast": 238,
        }

    def test_a_split_row_becomes_b_a_b_a_b(self) -> None:
        row = split_dialogue_row(self._row(), backchannel="ええ。")
        self.assertEqual([turn["speaker"] for turn in row["turns"]], ["B", "A", "B", "A", "B"])
        self.assertEqual(
            [turn["role"] for turn in row["turns"]],
            ["open", "body", "backchannel", "body", "close"],
        )
        self.assertEqual(row["turns"][2]["text"], "ええ。")

    def test_the_two_a_turns_rejoin_into_the_corpus_sentence(self) -> None:
        row = split_dialogue_row(self._row(), backchannel="ええ。")
        joined = "".join(turn["text"] for turn in row["turns"] if turn["speaker"] == "A")
        self.assertEqual(joined, self.SENTENCE)

    def test_the_other_fields_are_carried_over_untouched(self) -> None:
        row = split_dialogue_row(self._row(), backchannel="ええ。")
        self.assertEqual(row["dialogue_id"], "v-003")
        self.assertEqual(row["split"], "train")
        self.assertEqual(row["source_artifact_id"], "tsukuyomi-corpus-v1:VOICEACTRESS100_003")
        # Only project_script_frames, which needs pyopenjtalk, may change this.
        self.assertEqual(row["projected_frames_fast"], 238)

    def test_no_backchannel_leaves_the_row_three_turns(self) -> None:
        row = split_dialogue_row(self._row(), backchannel=None)
        self.assertEqual([turn["speaker"] for turn in row["turns"]], ["B", "A", "B"])
        self.assertEqual([turn["role"] for turn in row["turns"]], ["open", "body", "close"])

    def test_a_sentence_that_will_not_split_stays_three_turns(self) -> None:
        source = self._row()
        source["turns"][1]["text"] = "クィーンズアベニューアルファに所属している。"
        row = split_dialogue_row(source, backchannel="ええ。")
        self.assertEqual([turn["speaker"] for turn in row["turns"]], ["B", "A", "B"])
        self.assertEqual(row["turns"][1]["text"], "クィーンズアベニューアルファに所属している。")

    def test_a_row_that_is_not_b_a_b_is_rejected(self) -> None:
        source = self._row()
        source["turns"] = source["turns"][:2]
        with self.assertRaises(ValueError):
            split_dialogue_row(source, backchannel="ええ。")


class SummariseStructureTests(unittest.TestCase):
    ROWS = [
        {
            "dialogue_id": "v-001",
            "turns": [
                {"speaker": "B", "text": "アイウ"},
                {"speaker": "A", "text": "カキ、"},
                {"speaker": "B", "text": "ええ。"},
                {"speaker": "A", "text": "クケコ"},
                {"speaker": "B", "text": "サシ"},
            ],
        },
        {
            "dialogue_id": "v-002",
            "turns": [
                {"speaker": "B", "text": "アイ"},
                {"speaker": "A", "text": "カキクケコ"},
                {"speaker": "B", "text": "サ"},
            ],
        },
    ]

    def test_it_counts_turns_and_a_turns(self) -> None:
        summary = summarise_structure(self.ROWS)
        self.assertEqual(summary["dialogues"], 2)
        self.assertEqual(summary["turns_total"], 8)
        self.assertEqual(summary["turns_per_dialogue"], 4.0)
        self.assertEqual(summary["a_turns_total"], 3)
        self.assertEqual(summary["a_turns_per_dialogue"], 1.5)
        self.assertEqual(summary["a_turns_histogram"], {"1": 1, "2": 1})
        self.assertEqual(summary["speaker_shapes"], {"B-A-B": 1, "B-A-B-A-B": 1})

    def test_a_characters_count_every_a_turn(self) -> None:
        # This total is the invariant that makes the rebuild a rebuild rather than a rewrite.
        self.assertEqual(summarise_structure(self.ROWS)["a_characters"], 3 + 3 + 5)

    def test_an_empty_script_set_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            summarise_structure([])


class SummariseSplitPointsTests(unittest.TestCase):
    def test_it_counts_the_character_in_front_of_the_cut(self) -> None:
        rows = [
            {
                "dialogue_id": "v-001",
                "turns": [
                    {"speaker": "B", "text": "はじめ"},
                    {"speaker": "A", "text": "これは、"},
                    {"speaker": "B", "text": "ええ。"},
                    {"speaker": "A", "text": "そうです。"},
                    {"speaker": "B", "text": "おわり"},
                ],
            },
            {
                "dialogue_id": "v-002",
                "turns": [
                    {"speaker": "B", "text": "はじめ"},
                    {"speaker": "A", "text": "パンと、"},
                    {"speaker": "B", "text": "はい。"},
                    {"speaker": "A", "text": "スープ。"},
                    {"speaker": "B", "text": "おわり"},
                ],
            },
        ]
        summary = summarise_split_points(rows)
        self.assertEqual(summary["splits"], 2)
        self.assertEqual(summary["preceding_character"], {"は": 1, "と": 1})
        # A cut in front of a coordination leaves the listener mid-list; those get named.
        self.assertEqual(summary["mid_coordination"]["count"], 1)
        self.assertEqual(summary["mid_coordination"]["dialogue_ids"], ["v-002"])

    def test_unsplit_rows_are_not_counted(self) -> None:
        rows = [
            {
                "dialogue_id": "v-011",
                "turns": [
                    {"speaker": "B", "text": "はじめ"},
                    {"speaker": "A", "text": "所属している。"},
                    {"speaker": "B", "text": "おわり"},
                ],
            }
        ]
        self.assertEqual(summarise_split_points(rows)["splits"], 0)


class ValidateScriptsTests(unittest.TestCase):
    CORPUS = {
        "tsukuyomi-corpus-v1:VOICEACTRESS100_001": "また、東寺のように、五大明王と呼ばれる、主要な明王の中央に配されることも多い。",
        "tsukuyomi-corpus-v1:VOICEACTRESS100_002": "ニューイングランド風は、牛乳をベースとした、白いクリームスープであり、ボストンクラムチャウダーとも呼ばれる。",
    }

    def _dialogue(self, artifact_id: str, **overrides) -> dict:
        row = {
            "dialogue_id": "v-001",
            "source_artifact_id": artifact_id,
            "turns": [
                {"speaker": "B", "text": "Bの前置き"},
                {"speaker": "A", "text": self.CORPUS[artifact_id]},
                {"speaker": "B", "text": "Bの受け"},
            ],
        }
        row.update(overrides)
        return row

    def _split_dialogue(self, artifact_id: str, **overrides) -> dict:
        return split_dialogue_row(self._dialogue(artifact_id, **overrides), backchannel="ええ。")

    def test_a_well_formed_pair_of_dialogues_validates(self) -> None:
        rows = [
            self._dialogue("tsukuyomi-corpus-v1:VOICEACTRESS100_001", dialogue_id="v-001"),
            self._dialogue("tsukuyomi-corpus-v1:VOICEACTRESS100_002", dialogue_id="v-002"),
        ]
        report = validate_scripts(rows, corpus_texts=self.CORPUS, eval_texts=set())
        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["rows"], 2)
        self.assertEqual(report["a_texts_match_corpus"], True)

    def test_a_five_turn_dialogue_validates(self) -> None:
        rows = [
            self._split_dialogue("tsukuyomi-corpus-v1:VOICEACTRESS100_001", dialogue_id="v-001"),
            self._split_dialogue("tsukuyomi-corpus-v1:VOICEACTRESS100_002", dialogue_id="v-002"),
        ]
        report = validate_scripts(rows, corpus_texts=self.CORPUS, eval_texts=set())
        self.assertEqual(report["status"], "pass")

    def test_the_two_shapes_may_be_mixed(self) -> None:
        # Two of the 72 train sentences have no comma and stay B-A-B; both shapes ship.
        rows = [
            self._split_dialogue("tsukuyomi-corpus-v1:VOICEACTRESS100_001", dialogue_id="v-001"),
            self._dialogue("tsukuyomi-corpus-v1:VOICEACTRESS100_002", dialogue_id="v-002"),
        ]
        self.assertEqual(
            validate_scripts(rows, corpus_texts=self.CORPUS, eval_texts=set())["status"], "pass"
        )

    def test_an_altered_a_turn_is_caught(self) -> None:
        # A must speak the corpus sentence verbatim; the recording is the audio.
        rows = [self._dialogue("tsukuyomi-corpus-v1:VOICEACTRESS100_001")]
        rows[0]["turns"][1]["text"] = "また、東寺のように、五大明王と呼ばれます。"
        rows.append(self._dialogue("tsukuyomi-corpus-v1:VOICEACTRESS100_002", dialogue_id="v-002"))
        report = validate_scripts(rows, corpus_texts=self.CORPUS, eval_texts=set())
        self.assertEqual(report["status"], "fail")
        self.assertIn("v-001", str(report["altered_a_turns"]))

    def test_fragments_that_do_not_rejoin_are_caught(self) -> None:
        # A split that dropped the comma would desynchronise text from the recording.
        rows = [
            self._split_dialogue("tsukuyomi-corpus-v1:VOICEACTRESS100_001", dialogue_id="v-001"),
            self._split_dialogue("tsukuyomi-corpus-v1:VOICEACTRESS100_002", dialogue_id="v-002"),
        ]
        rows[0]["turns"][1]["text"] = rows[0]["turns"][1]["text"].rstrip("、")
        report = validate_scripts(rows, corpus_texts=self.CORPUS, eval_texts=set())
        self.assertEqual(report["status"], "fail")
        self.assertIn("v-001", str(report["altered_a_turns"]))

    def test_a_missing_corpus_sentence_fails_set_equality(self) -> None:
        # A count check would pass here; only set equality catches a duplicate plus a gap.
        rows = [
            self._dialogue("tsukuyomi-corpus-v1:VOICEACTRESS100_001", dialogue_id="v-001"),
            self._dialogue("tsukuyomi-corpus-v1:VOICEACTRESS100_001", dialogue_id="v-002"),
        ]
        report = validate_scripts(rows, corpus_texts=self.CORPUS, eval_texts=set())
        self.assertEqual(report["status"], "fail")
        self.assertFalse(report["a_texts_match_corpus"])

    def test_speaker_order_must_be_one_of_the_agreed_shapes(self) -> None:
        rows = [self._dialogue("tsukuyomi-corpus-v1:VOICEACTRESS100_001")]
        rows[0]["turns"] = [{"speaker": "A", "text": list(self.CORPUS.values())[0]}]
        rows.append(self._dialogue("tsukuyomi-corpus-v1:VOICEACTRESS100_002", dialogue_id="v-002"))
        report = validate_scripts(rows, corpus_texts=self.CORPUS, eval_texts=set())
        self.assertEqual(report["status"], "fail")
        self.assertIn("v-001", str(report["bad_turn_structure"]))

    def test_a_dialogue_that_ends_on_a_is_rejected(self) -> None:
        rows = [self._split_dialogue("tsukuyomi-corpus-v1:VOICEACTRESS100_001")]
        rows[0]["turns"] = rows[0]["turns"][:4]
        rows.append(self._dialogue("tsukuyomi-corpus-v1:VOICEACTRESS100_002", dialogue_id="v-002"))
        report = validate_scripts(rows, corpus_texts=self.CORPUS, eval_texts=set())
        self.assertEqual(report["status"], "fail")
        self.assertIn("v-001", str(report["bad_turn_structure"]))

    def test_reusing_an_evaluation_sentence_is_caught(self) -> None:
        rows = [
            self._dialogue("tsukuyomi-corpus-v1:VOICEACTRESS100_001", dialogue_id="v-001"),
            self._dialogue("tsukuyomi-corpus-v1:VOICEACTRESS100_002", dialogue_id="v-002"),
        ]
        rows[0]["turns"][0]["text"] = "評価用に固定した文です"
        report = validate_scripts(
            rows, corpus_texts=self.CORPUS, eval_texts={"評価用に固定した文です"}
        )
        self.assertEqual(report["status"], "fail")
        self.assertEqual(report["eval_overlap_count"], 1)

    def test_a_held_out_sentence_spoken_by_a_is_caught(self) -> None:
        # The evaluation set is worth what its separation from training is worth.
        corpus = dict(self.CORPUS)
        corpus["tsukuyomi-corpus-v1:VOICEACTRESS100_099"] = "評価用に取り置いた文です。"
        rows = [
            self._dialogue("tsukuyomi-corpus-v1:VOICEACTRESS100_001", dialogue_id="v-001"),
            self._dialogue("tsukuyomi-corpus-v1:VOICEACTRESS100_002", dialogue_id="v-002"),
            {
                "dialogue_id": "v-003",
                "source_artifact_id": "tsukuyomi-corpus-v1:VOICEACTRESS100_099",
                "turns": [
                    {"speaker": "B", "text": "Bの前置き"},
                    {"speaker": "A", "text": "評価用に取り置いた文です。"},
                    {"speaker": "B", "text": "Bの受け"},
                ],
            },
        ]
        report = validate_scripts(
            rows,
            corpus_texts=corpus,
            eval_texts=set(),
            held_out_texts={"評価用に取り置いた文です。"},
        )
        self.assertEqual(report["status"], "fail")
        self.assertEqual(report["held_out_overlap_count"], 1)

    def test_a_held_out_sentence_cut_in_two_is_still_caught(self) -> None:
        # A per-turn check would miss this; the joined A text is compared as well.
        corpus = dict(self.CORPUS)
        corpus["tsukuyomi-corpus-v1:VOICEACTRESS100_099"] = "評価用に、取り置いた文です。"
        rows = [
            self._dialogue("tsukuyomi-corpus-v1:VOICEACTRESS100_001", dialogue_id="v-001"),
            self._dialogue("tsukuyomi-corpus-v1:VOICEACTRESS100_002", dialogue_id="v-002"),
            {
                "dialogue_id": "v-003",
                "source_artifact_id": "tsukuyomi-corpus-v1:VOICEACTRESS100_099",
                "turns": [
                    {"speaker": "B", "text": "Bの前置き"},
                    {"speaker": "A", "text": "評価用に、"},
                    {"speaker": "B", "text": "ええ。"},
                    {"speaker": "A", "text": "取り置いた文です。"},
                    {"speaker": "B", "text": "Bの受け"},
                ],
            },
        ]
        report = validate_scripts(
            rows,
            corpus_texts=corpus,
            eval_texts=set(),
            held_out_texts={"評価用に、取り置いた文です。"},
        )
        self.assertEqual(report["status"], "fail")
        self.assertEqual(report["held_out_overlap_count"], 1)

    def test_an_a_turn_under_the_floor_is_caught_when_a_floor_is_given(self) -> None:
        rows = [
            self._split_dialogue("tsukuyomi-corpus-v1:VOICEACTRESS100_001", dialogue_id="v-001"),
            self._split_dialogue("tsukuyomi-corpus-v1:VOICEACTRESS100_002", dialogue_id="v-002"),
        ]
        report = validate_scripts(
            rows, corpus_texts=self.CORPUS, eval_texts=set(), min_a_turn_chars=100
        )
        self.assertEqual(report["status"], "fail")
        self.assertEqual(len(report["short_a_turns"]), 4)

    def test_duplicate_dialogue_ids_are_caught(self) -> None:
        rows = [
            self._dialogue("tsukuyomi-corpus-v1:VOICEACTRESS100_001", dialogue_id="v-001"),
            self._dialogue("tsukuyomi-corpus-v1:VOICEACTRESS100_002", dialogue_id="v-001"),
        ]
        report = validate_scripts(rows, corpus_texts=self.CORPUS, eval_texts=set())
        self.assertEqual(report["status"], "fail")
        self.assertIn("v-001", report["duplicate_dialogue_ids"])

    def test_an_empty_b_turn_is_caught(self) -> None:
        rows = [
            self._dialogue("tsukuyomi-corpus-v1:VOICEACTRESS100_001", dialogue_id="v-001"),
            self._dialogue("tsukuyomi-corpus-v1:VOICEACTRESS100_002", dialogue_id="v-002"),
        ]
        rows[0]["turns"][0]["text"] = "   "
        report = validate_scripts(rows, corpus_texts=self.CORPUS, eval_texts=set())
        self.assertEqual(report["status"], "fail")
        self.assertIn("v-001", str(report["empty_turns"]))

    def test_an_empty_backchannel_is_caught(self) -> None:
        rows = [
            self._split_dialogue("tsukuyomi-corpus-v1:VOICEACTRESS100_001", dialogue_id="v-001"),
            self._split_dialogue("tsukuyomi-corpus-v1:VOICEACTRESS100_002", dialogue_id="v-002"),
        ]
        rows[0]["turns"][2]["text"] = ""
        report = validate_scripts(rows, corpus_texts=self.CORPUS, eval_texts=set())
        self.assertEqual(report["status"], "fail")
        self.assertIn("v-001#2", report["empty_turns"])


class BuildV2ScriptsTests(unittest.TestCase):
    def _rows(self) -> list[dict]:
        return [
            {
                "schema_version": 1,
                "dialogue_id": "v-001",
                "source_artifact_id": "corpus:001",
                "split": "train",
                "turns": [
                    {"speaker": "B", "text": "おたずねします。"},
                    {"speaker": "A", "text": "これは前半でして、こちらが後半です。"},
                    {"speaker": "B", "text": "よくわかりました。"},
                ],
                "projected_frames_fast": 240,
            },
            {
                "schema_version": 1,
                "dialogue_id": "v-002",
                "source_artifact_id": "corpus:002",
                "split": "dev",
                "turns": [
                    {"speaker": "B", "text": "おたずねします。"},
                    {"speaker": "A", "text": "読点のない一文です。"},
                    {"speaker": "B", "text": "よくわかりました。"},
                ],
                "projected_frames_fast": 220,
            },
        ]

    def test_it_splits_what_it_can_and_leaves_what_it_cannot(self) -> None:
        built = build_v2_scripts(self._rows(), seed=20260825)
        shapes = ["-".join(turn["speaker"] for turn in row["turns"]) for row in built["rows"]]
        self.assertEqual(shapes, ["B-A-B-A-B", "B-A-B"])
        self.assertEqual(len(built["seams"]), 1)
        self.assertEqual(built["unsplit"][0]["dialogue_id"], "v-002")
        self.assertEqual(built["unsplit"][0]["reason"], "no-comma")

    def test_the_a_text_survives_the_rebuild(self) -> None:
        source = self._rows()
        built = build_v2_scripts(source, seed=20260825)
        for before, after in zip(source, built["rows"], strict=True):
            joined = "".join(turn["text"] for turn in after["turns"] if turn["speaker"] == "A")
            self.assertEqual(joined, before["turns"][1]["text"])

    def test_the_same_seed_rebuilds_the_same_file(self) -> None:
        first = build_v2_scripts(self._rows(), seed=20260825)
        second = build_v2_scripts(self._rows(), seed=20260825)
        self.assertEqual(first["rows"], second["rows"])

    def test_a_row_that_is_not_b_a_b_stops_the_build(self) -> None:
        rows = self._rows()
        rows[0]["turns"].append({"speaker": "A", "text": "余分な turn です。"})
        with self.assertRaises(ValueError):
            build_v2_scripts(rows, seed=20260825)

    def test_a_fragment_under_the_floor_is_reported_with_its_reason(self) -> None:
        rows = self._rows()
        rows[0]["turns"][1]["text"] = "ああ、いいいいいいいいいい。"
        built = build_v2_scripts(rows, seed=20260825)
        self.assertEqual(
            built["unsplit"][0]["reason"],
            "central-comma-leaves-a-fragment-under-the-floor",
        )


class CollectEvalTextsTests(unittest.TestCase):
    def test_it_collects_every_human_readable_field(self) -> None:
        rows = [
            {
                "id": "style-01",
                "prompt": "問いです。",
                "preferred": "こちら。",
                "dispreferred": "あちら。",
            },
            {"id": "tts-01", "text": "読み上げ文。", "tags": ["number"]},
        ]
        self.assertEqual(
            collect_eval_texts(rows), {"問いです。", "こちら。", "あちら。", "読み上げ文。"}
        )

    def test_blank_and_non_string_fields_are_ignored(self) -> None:
        self.assertEqual(collect_eval_texts([{"text": "   ", "prompt": None}]), set())


class ShippedScriptsTests(unittest.TestCase):
    """The committed M3-R scripts, checked against the M3 scripts they were rebuilt from.

    These read the artifacts rather than a fixture on purpose: the gates the parent plan
    states - A's characters unchanged, two A turns per dialogue, no evaluation sentence in
    training - are properties of the shipped file, and a unit test over a fixture would
    keep passing after somebody edited the file by hand.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.v1 = _read_jsonl(EXPERIMENT_ROOT / "m3" / "scripts" / "dialogues-v1.jsonl")
        cls.v2 = _read_jsonl(EXPERIMENT_ROOT / "m3r" / "scripts" / "dialogues-v2.jsonl")
        cls.corpus = _read_jsonl(EXPERIMENT_ROOT / "manifests" / "tsukuyomi-corpus-v1.jsonl")
        cls.split_map_v1 = json.loads(
            (EXPERIMENT_ROOT / "m3" / "scripts" / "split-map-v1.json").read_text(encoding="utf-8")
        )
        cls.split_map_v2 = json.loads(
            (EXPERIMENT_ROOT / "m3r" / "scripts" / "split-map-v2.json").read_text(encoding="utf-8")
        )
        cls.report = json.loads(
            (EXPERIMENT_ROOT / "reports" / "m3r-script-validation.json").read_text(encoding="utf-8")
        )

    def test_every_dialogue_survives_the_rebuild(self) -> None:
        # Losing a dialogue would drop the step count below M3's 45, which is the one thing
        # splitting rather than pairing was chosen to protect.
        self.assertEqual(
            [row["dialogue_id"] for row in self.v2], [row["dialogue_id"] for row in self.v1]
        )

    def test_a_says_exactly_what_it_said_in_v1(self) -> None:
        for before, after in zip(self.v1, self.v2, strict=True):
            joined = "".join(turn["text"] for turn in after["turns"] if turn["speaker"] == "A")
            self.assertEqual(joined, before["turns"][1]["text"], before["dialogue_id"])

    def test_the_total_a_characters_are_unchanged(self) -> None:
        self.assertEqual(
            summarise_structure(self.v2)["a_characters"],
            summarise_structure(self.v1)["a_characters"],
        )

    def test_a_speaks_twice_in_every_dialogue_that_could_be_split(self) -> None:
        summary = summarise_structure(self.v2)
        self.assertEqual(summary["a_turns_histogram"], {"1": 2, "2": 78})
        self.assertEqual(summary["speaker_shapes"], {"B-A-B": 2, "B-A-B-A-B": 78})
        self.assertGreaterEqual(summary["turns_per_dialogue"], 4.9)
        self.assertGreaterEqual(summary["a_turns_per_dialogue"], 1.9)

    def test_the_train_split_still_makes_45_steps(self) -> None:
        train = [row for row in self.v2 if row["split"] == "train"]
        self.assertEqual(len(train), 72)
        self.assertEqual(len(train) // 8 * 5, 45)

    def test_the_split_assignment_is_inherited_unchanged(self) -> None:
        # A dialogue that moved between train and dev would make M3-R and M3 incomparable
        # for a reason that has nothing to do with the rebuild.
        self.assertEqual(self.split_map_v2["assignment"], self.split_map_v1["assignment"])
        self.assertEqual(self.split_map_v2["counts"]["train"], 72)
        self.assertEqual(self.split_map_v2["counts"]["dev"], 8)
        for row in self.v2:
            self.assertEqual(row["split"], self.split_map_v2["assignment"][row["dialogue_id"]])

    def test_every_backchannel_comes_from_the_committed_pool(self) -> None:
        for row in self.v2:
            if len(row["turns"]) != 5:
                continue
            self.assertEqual(row["turns"][2]["speaker"], "B")
            self.assertEqual(row["turns"][2]["role"], "backchannel")
            self.assertIn(row["turns"][2]["text"], BACKCHANNEL_POOL)

    def test_no_backchannel_takes_more_than_its_share(self) -> None:
        used = [row["turns"][2]["text"] for row in self.v2 if len(row["turns"]) == 5]
        counts = [used.count(phrase) for phrase in BACKCHANNEL_POOL]
        self.assertLessEqual(max(counts) - min(counts), 1)

    def test_no_a_fragment_is_under_the_floor(self) -> None:
        for row in self.v2:
            if len(row["turns"]) != 5:
                continue
            for index in (1, 3):
                self.assertGreaterEqual(
                    len(row["turns"][index]["text"]), MIN_FRAGMENT_CHARS, row["dialogue_id"]
                )

    def test_the_scripts_validate_against_the_corpus_and_the_evaluation_sets(self) -> None:
        eval_texts: set[str] = set()
        for path in sorted((EXPERIMENT_ROOT / "eval").glob("*.jsonl")):
            eval_texts |= collect_eval_texts(_read_jsonl(path))
        report = validate_scripts(
            self.v2,
            corpus_texts={
                row["artifact_id"]: row["text"] for row in self.corpus if row["split"] == "train"
            },
            eval_texts=eval_texts,
            held_out_texts=[row["text"] for row in self.corpus if row["split"] in {"dev", "test"}],
            min_a_turn_chars=MIN_FRAGMENT_CHARS,
        )
        self.assertEqual(report["status"], "pass", report)
        self.assertEqual(report["eval_overlap_count"], 0)
        self.assertEqual(report["held_out_overlap_count"], 0)

    def test_the_validation_report_describes_the_file_that_shipped(self) -> None:
        path = REPOSITORY_ROOT / self.report["artifact"]["path"]
        self.assertEqual(path.stat().st_size, self.report["artifact"]["byte_size"])
        self.assertEqual(
            hashlib.sha256(path.read_bytes()).hexdigest(), self.report["artifact"]["sha256"]
        )
        self.assertEqual(self.report["checks"]["status"], "pass")
        self.assertTrue(self.report["invariants"]["a_characters_unchanged"])
        self.assertEqual(self.report["length"]["below_floor"], 0)


if __name__ == "__main__":
    unittest.main()
