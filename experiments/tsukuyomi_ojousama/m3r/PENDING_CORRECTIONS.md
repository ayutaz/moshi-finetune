# 適用待ちの修正 2 件

作成日: 2026-08-26

この 2 件は**内容が確定しており、置き換える本文もここにある**。適用できなかったのは
判断が付かなかったからではなく、作業セッションの途中でこの 2 ファイルが読み書きできなく
なったためである。**適用したらこのファイルを削除する。**

---

## 1. `TOKENIZE_COMMANDS.md` が出荷物と違う `dataset_id` を書いている

### 症状

`TOKENIZE_COMMANDS.md` §4（`record-tokenize` の節）が

```
--dataset_id   v-real-r1
--manifest     experiments/tsukuyomi_ojousama/manifests/v-real-r1.jsonl
--out          experiments/tsukuyomi_ojousama/manifests/v-real-r1-tokenize.json
```

と書いている。**出荷物は `v-real-v2` である**（`m3r/scripts/split-map-v2.json`、
`manifests/v-real-v2.jsonl`、`manifests/v-real-v2-tokenize.json`）。
手順どおり実行すると別の `dataset_id` の sidecar ができ、
`tests/test_experiment_assets.py::TokenizeFlagRecordTests` は
`v-real-v2.jsonl` に sidecar が無いとして落ちる。

`v-real-r1` はリポジトリ内で他に 1 箇所、`reports/m3r-tokenize-fix.json:437` の
再現コマンド例にも残っている。そちらは**過去の実行の記録**なので書き換えない。

### 同じ節で直したその他

| 箇所 | 旧 | 新 | 理由 |
| --- | --- | --- | --- |
| §4 `--recorded_at` | `2026-08-25` | `2026-08-26` | 出荷 sidecar の値 |
| §6 `--audio_chunk_size` の注記 | 「M3-R の 60 秒超でも」 | 「M3-R の 1 行は 19〜24 秒」 | 系列長 60 秒は[監査](../../../docs/experiments/j-moshi-tsukuyomi-ojousama-m3r-dataset-audit.md) §2.5 で**撤回済み** |
| §4 | tokenize_text のみ記録 | audio / text / parquet を 3 回記録 | 下の 2 件目と対 |
| §6 の表 | フラグ名のみ | ツール列を追加し audio / parquet も掲載 | `--device` が表に無かった |

### 適用方法

[`TOKENIZE_COMMANDS.replacement.md`](./TOKENIZE_COMMANDS.replacement.md) の
**冒頭 6 行のバナーを除いた本文**で `TOKENIZE_COMMANDS.md` を置き換え、
`TOKENIZE_COMMANDS.replacement.md` を削除する。

```bash
cd /Users/inamotoyuuta/Desktop/moshi-finetune/experiments/tsukuyomi_ojousama/m3r
tail -n +7 TOKENIZE_COMMANDS.replacement.md > TOKENIZE_COMMANDS.md
rm TOKENIZE_COMMANDS.replacement.md
rm PENDING_CORRECTIONS.md   # 2 件目も済んでいれば
```

置き換え後、`DATASET_BUILD.md` と本ファイルからの参照が切れないか確認すること
（`DATASET_BUILD.md` は `TOKENIZE_COMMANDS.md` を参照しており、そちらは変わらない）。

---

## 2. `tests/test_experiment_assets.py` の flag 検査が 1 ツールを前提にしている

### 症状

`TokenizeFlagRecordTests::test_every_invocation_states_every_flag_explicitly` は

```python
self.assertEqual(
    set(flags),
    set(TOKENIZE_TEXT_FLAGS),
    f"{sidecar.name}: the flag record is incomplete",
)
```

を **sidecar の全 invocation** に対して行う。sidecar が `tokenize_text` だけを持っていた
間は正しかった。`tools/tokenize_flags.py` が 3 ツールを記録できるようにしたので、
`tokenize_audio` の invocation を 1 本書いた瞬間にこの assert が落ちる。

**落ち方が悪い。** 記録がより完全になったところで赤になるので、
一番手近な直し方が「記録を元に戻す」になる。

### 直し方

`tools/tokenize_flags.check_invocation` に委ねる。invocation の `tool` を見て
そのツールの表と突き合わせ、問題を文で返す（`tool` が無い過去の invocation は
`tokenize_text` として読む）。

```python
# 差し替え前
from tools.text_stream_audit import TOKENIZE_TEXT_FLAGS
...
                flags = invocation.get("flags", {})
                self.assertEqual(
                    set(flags),
                    set(TOKENIZE_TEXT_FLAGS),
                    f"{sidecar.name}: the flag record is incomplete",
                )
                self.assertIsInstance(
                    flags["no_whitespace_before_word"],
                    bool,
                    f"{sidecar.name}: the whitespace flag is not a stated boolean",
                )

# 差し替え後
from tools.tokenize_flags import TOKENIZE_TEXT, check_invocation, invocation_tool
...
                self.assertEqual(
                    check_invocation(invocation, where=sidecar.name),
                    [],
                    f"{sidecar.name}: the flag record is incomplete",
                )
                if invocation_tool(invocation) == TOKENIZE_TEXT.name:
                    self.assertIsInstance(
                        invocation["flags"]["no_whitespace_before_word"],
                        bool,
                        f"{sidecar.name}: the whitespace flag is not a stated boolean",
                    )
```

`test_dropping_the_whitespace_flag_requires_a_written_defect` も
`invocation["flags"]["no_whitespace_before_word"]` を全 invocation に対して引くので、
`tokenize_text` の invocation だけを見るように絞る必要がある。

```python
            dropped = [
                invocation
                for invocation in record["invocations"]
                if invocation_tool(invocation) == TOKENIZE_TEXT.name
                and invocation["flags"]["no_whitespace_before_word"] is False
            ]
```

### 気づく仕組みは既に入っている

`tests/test_tokenize_flags.py::AssetGateCompatibilityTests` が、
**多ツール sidecar が存在し、かつ上の一般化がまだのとき**に落ちる。
今日の状態（多ツール sidecar 0 件）では通る。negative control は確認済み
—— 3 ツールの sidecar を 1 本置くと、直すべき箇所を名指しして落ちる。

---

## 3. まだ閉じていないこと（この 2 件とは別）

`manifests/v-real-v2-tokenize.json` には `tokenize_text` の invocation しか入っていない。
`--device cpu` を含む audio / parquet のフラグは
`reports/m3r-tokenize.json` にしか無く、**manifest からは辿れない**。

閉じるには `TOKENIZE_COMMANDS.md` §4 の 6 本（3 ツール × 2 split）を実行し、
`tests/test_tokenize_flags.py` の `PRE_MECHANISM_SIDECARS` から
`v-real-v2-tokenize.json` の行を消す。sidecar は `manifests/` 配下にあり、
本作業の担当範囲外だったため実行していない。
