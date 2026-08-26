# M3-R dataset の作り直し手順 — split 材料化から manifest・検証まで

作成日: 2026-08-26
対象: `v-real-v2`（`experiments/tsukuyomi_ojousama/m3r/v-real-v2-dataset.json` が定義する）
tokenize 本体のフラグは [`TOKENIZE_COMMANDS.md`](./TOKENIZE_COMMANDS.md) が正とする。

この文書は **stereo が出来てから manifest が出来るまで**をコピーして実行できる形で持つ。

> **先に [`PENDING_CORRECTIONS.md`](./PENDING_CORRECTIONS.md) を読むこと。**
> `TOKENIZE_COMMANDS.md` §4 は `--dataset_id v-real-r1` と書いているが、
> **出荷物は `v-real-v2`** である。置き換える本文は
> [`TOKENIZE_COMMANDS.replacement.md`](./TOKENIZE_COMMANDS.replacement.md) にある。
> 適用するまで、§4 の `dataset_id` は読み替えて実行すること。

---

## 0. なぜこの文書があるか

`v-real-v2` は `data/experiments/tsukuyomi_ojousama/m3r/v-real/build-scripts/` の 6 本の
スクリプトで作られた。**`data/` は gitignore なので、その 6 本はコミットされない。**
結果、manifest を作り直す手順がリポジトリに残らず、そこに入った純粋ロジック
（sequence 行の照合、manifest 行の組み立て、held-out 照合）はテストもされなかった。

**これは M3 で `--no_whitespace_before_word` が黙って落ちたのと同じ種類の欠陥である。**
実行したことが成果物に残らない、という一段上の形をしている。

そこで純粋ロジックを `tools/` に移し、この文書を手順の正本にした。

| 元スクリプト | 移した先 |
| --- | --- |
| `make_splits.py` | `tools/dialogue_manifest.py` の `sequence_plan` / `materialise_splits` |
| `build_manifest.py` | `tools/dialogue_manifest.py` の `manifest_row` / `build_rows` |
| `verify_parquet.py` | `tools/parquet_shape.py` と、既にあった `tools/text_stream_audit.py audit` / `tools/training_shape.py` |
| `heldout_check.py` | `tools/dataset_leakage.py`（id 照合・本文照合） |
| `final_checks.py` | `tools/dataset_leakage.py`（語 transcript・parquet 復号・台本一致） |
| `build_report.py` の command 組み立て | `tools/tokenize_flags.py` の `render_command` |

データセットの権利表示（credit / license / redistribution）と入力パスは
**`m3r/v-real-v2-dataset.json`** が持つ。`tools/` のソースには埋めない。
埋めると、権利に関わる文字列の変更が差分に出ずにレビューを素通りする。

---

## 1. 前提

```bash
cd /Users/inamotoyuuta/Desktop/moshi-finetune
D=data/experiments/tsukuyomi_ojousama/m3r          # M3-R の dataset root
DS=v-real                                          # M3-R は V-real のみ
SPEC=experiments/tsukuyomi_ojousama/m3r/v-real-v2-dataset.json
TOK=$(ls ~/.cache/huggingface/hub/models--nu-dialogue--j-moshi-ext/snapshots/*/tokenizer_spm_32k_3.model)
NICE="nice -n 19"                                  # このマシンの負荷を上げない
ENV2="OMP_NUM_THREADS=2 MKL_NUM_THREADS=2"
```

`uv` は `--no-project` で走らせる。`uv.lock` は gitignore なので、依存はコマンドに書き切る。

---

## 2. split ディレクトリを作る

`sequences/<split>/` の行を **対話の名前で** `<split>/{audio,text}` にコピーする。
名前が `dialogue_id` の名前空間になる（`train/v-001`。`train/train-seq-001` ではない）。
この名前空間が parquet と manifest と M3 の行を結ぶので、命名は装飾ではない。

```bash
$NICE uv run --python 3.12 --no-project python -m tools.dialogue_manifest materialise-splits \
  --dataset_root  "$D/$DS" \
  --build_report  experiments/tsukuyomi_ojousama/reports/m3r-timeline.json \
  --out           "$D/$DS/split-copy-checks.json"
```

- `group_size` が 1 でなければ**落ちる**。1 行に 4 対話入っていたら、その行に取るべき
  対話名が 1 つに決まらない。
- 全コピーは **3 者照合**を通る: sequence ファイル / 対話ファイル / ビルド報告の sha256。
  1 つでも食い違えば**何も書かずに**落ちる。半分だけ材料化された状態を残さない。
- `--dry_run` を付けると照合だけして書かない。

期待値: `verified_triples 160`（80 対話 × audio/text）。

---

## 3. tokenize する

[`TOKENIZE_COMMANDS.md`](./TOKENIZE_COMMANDS.md) の §1〜§3 をその順で実行する。
**audio → text → parquet。** `--device cpu`。

実行したら §4 の `record-tokenize` を **3 回**（tool ごとに）× split ごとに走らせる。
これを飛ばすと `--device` が manifest から辿れなくなる。

---

## 4. manifest を書く

```bash
$NICE uv run --python 3.12 --no-project python -m tools.dialogue_manifest build \
  --dataset "$SPEC"
```

- checksum は**すべてディスクから読む**。前の報告から写さない。
- 行ごとに 3 者が split を名乗る（split-map / 台本 / ビルド報告）。食い違えば落ちる。
- wav の sha256 がビルド報告と違えば落ちる。報告が見ていないファイルの checksum を
  manifest に書かないため。
- 出力は `--out` 省略時 spec の `manifest`（= `manifests/v-real-v2.jsonl`）。

続けて既存の validator を通す。

```bash
$NICE uv run --python 3.12 --no-project python -c "
import json, sys
from pathlib import Path
from tools.experiment_data import validate_manifest
rows = [json.loads(l) for l in Path('experiments/tsukuyomi_ojousama/manifests/v-real-v2.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]
print(json.dumps(validate_manifest(rows, data_root=Path('data/experiments/tsukuyomi_ojousama')), ensure_ascii=False, indent=2))
"
```

> **validator が見ていない枝がある。** `tools/experiment_data.validate_manifest` の
> audio metadata 照合は `media_type == "audio/wav"` でしか発火せず、本リポジトリの
> manifest はすべて `audio/x-wav` である。`tools/dialogue_manifest.audio_metadata` が
> `_audio_metadata` と同じ丸め（6 桁）で書いているのはそのためで、
> **照合そのものは走っていない**。指した行を黙って飛ばす validator は、
> 誰も書き留めなかったフラグと同じ形の失敗である。

---

## 5. 検証

### 5-1. parquet の形（trainer が受け取る stream 数）

```bash
$NICE uv run --python 3.12 --no-project --with numpy --with pandas --with pyarrow \
  python -m tools.parquet_shape measure \
    "$D/$DS/parquet/train-001-of-001.parquet" \
    "$D/$DS/parquet/dev-001-of-001.parquet" \
    --out experiments/tsukuyomi_ojousama/reports/m3r-parquet-shape.json
```

`utils/data.main_speaker_streams`（trainer 自身の関数）で数える。**期待値 17**。
speaker 列の `[9, T]` を見て 17 と言うのは推論であって測定ではない。

### 5-2. text stream（裸の U+2581）

`TOKENIZE_COMMANDS.md` §5 の `text_stream_audit audit`。**ゲートは 0.1、期待値 0.0。**

### 5-3. held-out の混入

```bash
$NICE uv run --python 3.12 --no-project --with numpy --with pandas --with pyarrow --with sentencepiece \
  python -m tools.dataset_leakage check \
    --corpus_manifest     experiments/tsukuyomi_ojousama/manifests/tsukuyomi-corpus-v1.jsonl \
    --dialogues           experiments/tsukuyomi_ojousama/m3r/scripts/dialogues-v2.jsonl \
    --held_out_dir        data/experiments/tsukuyomi_ojousama/baseline-input/tsukuyomi-heldout \
    --word_transcript_dir "$D/$DS/train/text" "$D/$DS/dev/text" \
    --scripts_agree \
    --parquet             "$D/$DS/parquet/train-001-of-001.parquet" \
                          "$D/$DS/parquet/dev-001-of-001.parquet" \
    --text_tokenizer_path "$TOK" \
    --out experiments/tsukuyomi_ojousama/reports/m3r-heldout.json
```

3 経路を独立に通す。片方だけでは足りない理由:

| 経路 | 何を捕まえる | 見えないもの |
| --- | --- | --- |
| artifact id | 台本が引用したコーパス文が dev/test でないか | 改名・再生成されたファイル |
| 語 transcript の本文 | tokenize にかけた文字列そのもの | transcript が台本と別物のとき |
| parquet の復号 text stream | **モデルが実際に読む列** | — |

判定は「まるごと一致が 0 件」。数値としては **最長共有ラン**も出る。boolean は
「漏れたか」しか答えないが、ラン長は「どこまで近づいたか」も答える。
ビルドを跨いで同じ値のままなら、それは `false` にはない証拠である。

> 正規化は `tools/experiment_data._normalise_text`（NFKC・casefold・英数字以外を除去）を使う。
> `tools/memorisation.py` が同じものを借りているのと同じ理由で、
> **「同じ文とは何か」で食い違う正規化器を 2 つ持たない。**
> 撤回した build-scripts は独自の正規化器を持っていたので、
> 最長ランの数値はそちらの報告（5 文字）と一致しない可能性がある。**測って記録すること。**

`--scripts_agree` は語 transcript を台本と**話者ごとに**照合する。対話単位で足し合わせると、
A の語が B のチャンネルに乗っている dataset が一致してしまう。

終了コードは合格 0 / 不合格 1。

### 5-4. 学習の形

```bash
$NICE uv run --python 3.12 --no-project python -c "
from tools.training_shape import global_batch_size, steps_per_epoch, total_steps, checkpoint_steps
b = global_batch_size(per_device=1, processes=2, gradient_accumulation=4)
s = steps_per_epoch(examples=72, batch=b)
print('global batch', b, '/ steps per epoch', s,
      '/ total', total_steps(examples=72, batch=b, epochs=5),
      '/ checkpoints', checkpoint_steps(examples=72, batch=b, epochs=5, save_steps=s))
"
```

期待値: global batch 8、steps/epoch 9、**total 45**（M3 と同一）、checkpoint 5 本。
`--examples` は parquet の train 行数を入れる。行数が変われば step が変わり、
M3 との比較が「schedule の違い」に汚染される。

---

## 6. テスト

```bash
uv run --python 3.12 --with pytest --no-sync python -m pytest tests -q
```

`tests/test_experiment_assets.py` がデータ台帳のゲートを持つ。落ちたら台帳が間違っている。

---

## 7. 実行順のまとめ

```
1. split 材料化 : dialogue_manifest materialise-splits   → $D/$DS/{train,dev}/{audio,text}
2. tokenize     : TOKENIZE_COMMANDS.md §1-3              → tok-audio / tok-text / parquet
3. フラグ記録   : text_stream_audit record-tokenize × 3 tool × 2 split
4. manifest     : dialogue_manifest build                → manifests/v-real-v2.jsonl
5. 検証         : parquet_shape / text_stream_audit audit / dataset_leakage / training_shape
6. テスト       : pytest tests -q
```

**3 を飛ばすと 2 が消える。** それが M3 で起きたことである。
