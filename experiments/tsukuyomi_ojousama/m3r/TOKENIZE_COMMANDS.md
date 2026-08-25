# M3-R tokenize コマンド — 全フラグ、実行順、落としたときに何が起きるか

作成日: 2026-08-25
確定した工程: [M3-R 計画](../../../docs/experiments/j-moshi-tsukuyomi-ojousama-m3r-plan.md) 第 2 段 2-1
実証記録: [`reports/m3r-tokenize-fix.json`](../reports/m3r-tokenize-fix.json)

この文書は **M3-R の dataset を作るときにそのまま実行するコマンド**を持つ。
すべて本セッションで実行し、出力を確認したものである。推定で書いた行はない。

M3 はコマンドラインをどこにも記録しなかった。`m3/tokenize.log` に残っていたのは
`tokenize_text rc=0` の 4 行だけで、`--no_whitespace_before_word` が落ちていたことは
**出荷から 3 日後に parquet を数え直して初めて分かった**。この文書はその再発を止めるためにある。

---

## 0. 前提

| 項目 | 値 | 理由 |
| --- | --- | --- |
| Python | 3.12 | プロジェクトの `.venv` は 3.13 で torch が入らない |
| 実行形式 | `uv run --no-project` | `uv.lock` は gitignore。依存はコマンドに書き切る |
| cwd | リポジトリ root | `python -m tools.*` が `tools` を見つけるため |
| text tokenizer | `nu-dialogue/j-moshi-ext` / `tokenizer_spm_32k_3.model` | 既定の `kyutai/moshiko-pytorch-bf16` は英語用。日本語が全部 byte-fallback になり**全対話が skip される** |
| audio tokenizer | `kyutai/moshiko-pytorch-bf16` / `tokenizer-e351c8d8-checkpoint125.safetensors` | Mimi。既定のままでよい |

tokenizer の実体（HF cache 済み、本セッションで確認）:

```
/Users/inamotoyuuta/.cache/huggingface/hub/models--nu-dialogue--j-moshi-ext/snapshots/c4676f5aae2155bb20c300dd6c71b6115f97ec1f/tokenizer_spm_32k_3.model
sha256 b5cbdfa8aa7c54c8c5af85b78c309c54a5f2749a20468bf6f60eee007fe6fec1
```

共通の環境変数（以下のコマンドはこれを前提にする）:

```bash
cd /Users/inamotoyuuta/Desktop/moshi-finetune
D=data/experiments/tsukuyomi_ojousama/m3r          # M3-R の dataset root
DS=v-real                                          # M3-R は V-real のみ（V-tts は腕から外れている）
TOK=$(ls ~/.cache/huggingface/hub/models--nu-dialogue--j-moshi-ext/snapshots/*/tokenizer_spm_32k_3.model)
```

---

## 1. 音声 tokenize（`tools/tokenize_audio.py`）

**text より先に走らせる。** `prepare_dataset` は audio の長さに text を切り詰める／詰めるので、
audio が正本である。

```bash
for part in train dev; do
  uv run --python 3.12 --no-project \
    --with "torch==2.4.1" --with "torchaudio==2.4.1" \
    --with moshi==0.1.0 --with numpy --with soundfile --with huggingface_hub --with tqdm \
    python -m tools.tokenize_audio \
      --audio_dir              "$D/$DS/$part/audio" \
      --output_dir             "$D/$DS/$part/tok-audio" \
      --audio_tokenizer_repo   kyutai/moshiko-pytorch-bf16 \
      --audio_tokenizer_name   tokenizer-e351c8d8-checkpoint125.safetensors \
      --audio_chunk_size       1200 \
      --device                 cpu \
      --num_workers            1
done
```

### `--device` — mps は使えるが、bit 一致しない

`--device mps` は動く。ただし **CPU と bit 一致しない**（本セッションで実測）。

| 比較 | ファイル一致 | token 一致率 |
| --- | ---: | ---: |
| このマシンの cpu vs M3 出荷の cpu | **5/5** | **1.000000** (21600/21600) |
| このマシンの mps vs M3 出荷の cpu | 4/5 | 0.999907 (21598/21600) |

食い違いは `v-001` 話者 B の frame 245 の 1 フレームだけで、**codebook 4 と 6**（残差側）である。
codebook 0（semantic）はこの標本では一致した。RVQ の量子化境界で丸めが分かれる典型であり、
**バグではないが再現性は失われる**。

速度差は小さい（tqdm 実測、21 秒対話 5 本: mps 約 3.2 秒 / cpu 約 5.9 秒。
80 対話に外挿して mps 約 51 秒 / cpu 約 94 秒）。

**したがって: 学習に投入する dataset は `--device cpu` で作る。** 40 秒の節約に
「checksum が合わない」を買う価値はない。`--device mps` は探索・プローブ用とし、
使った場合は device を記録に残す（同じ wav から別の parquet が出る）。

`--device cuda` は GPU が見えないと `NoAcceleratorError` で落ちる。CPU に黙って落ちない
設計なので、ローカルでは `cpu` か `mps` を**名指しする**。

### `--num_workers`

`mps` と `cuda:N` は `resolve_worker_count` が 1 に固定する。`cpu` は指定どおり増える。
出力 .npz は対話ごとに独立なので、worker 数を変えても**中身は変わらない**。変わるのは
ログの順序と、`[skip]` が出たときの切り分けやすさである。M3 は 1 で走らせた。
**M3 と比較する以上、1 のままにする。**

---

## 2. text tokenize（`tools/tokenize_text.py`）

```bash
for part in train dev; do
  uv run --python 3.12 --no-project \
    --with numpy --with sentencepiece --with huggingface_hub --with tqdm \
    python -m tools.tokenize_text \
      --word_transcript_dir       "$D/$DS/$part/text" \
      --output_dir                "$D/$DS/$part/tok-text" \
      --text_tokenizer_repo       nu-dialogue/j-moshi-ext \
      --text_tokenizer_name       tokenizer_spm_32k_3.model \
      --no_whitespace_before_word \
      --text_padding_id           3 \
      --end_of_text_padding_id    0 \
      --audio_tokenizer_frame_rate 12.5 \
      --num_workers               1
done
```

上の**全フラグ明示形**が、既定に任せた形と **72/72 バイト一致**することを確認済み
（`reports/m3r-tokenize-fix.json` の `explicit_flags_equivalence`）。
明示しても挙動は変わらない。変わるのは、記録に残るかどうかである。

### `--no_whitespace_before_word` — 落としてはならない

**根拠は 2 箇所でリポジトリ自身が書いている。**

- `README-ja.md:109`「日本語や中国語など，単語間にスペースがない言語の場合 `--no_whitespace_before_word` フラグを使用してください」
- `README.md:112` 同旨

このフラグがないと `tokenize_text.py:59-64` が pyopenjtalk の切った全語の先頭に半角空白を足し、
SentencePiece がそれを 1 語ずつ裸の `▁`（id 9）として吐く。日本語では語の結合が壊れる。

M3 の実測（本セッションで parquet を作り直して確認）:

| | フラグなし（M3 出荷） | フラグあり |
| --- | ---: | ---: |
| 話者 A の text token | 3,172 | **1,356** |
| うち裸 `▁` | 1,416 (**44.64%**) | **0 (0.00%)** |
| 話者 B の text token | 7,769 | **2,910** |
| うち裸 `▁` | 3,621 (**46.61%**) | **0 (0.00%)** |

`v-001` 話者 A の text stream（`_` が U+2581、`<unk>` は end-of-text-padding id 0）:

```
フラグなし: <unk>_また<unk>_、_東<unk>寺<unk>_の_よう<unk>_に_、_五大<unk>_<unk>明<unk>王<unk>_と_呼ば_れる<unk>_、…
フラグあり: <unk>_また<unk>、東<unk>寺<unk>のように<unk>、五<unk>大<unk>明<unk>王<unk>と呼ばれる<unk>、…
```

「呼ば_れる」が「と呼ばれる」に戻る。非 pad フレームは 74 → 37 に減る。

### `--text_padding_id 3` / `--end_of_text_padding_id 0`

j-moshi-ext の tokenizer では id 3 = `[PAD]`、id 0 = `<unk>`。
既定と同じだが、**tokenizer を替えたら替わりうる値**なので明示する（`README-ja.md:109` が同じ注意を書いている）。

### `--audio_tokenizer_frame_rate 12.5`

Mimi は 12.5 Hz。**`int` ではなく `float`** である。
`tokenize_text.py` の該当箇所には、かつて `type=int` だったときに
`12` が黙って通り、text stream が 2 秒ごとに 1 フレームずつ音声からずれる—
どの成果物にも現れない—危険があった旨のコメントが残っている。今は `type=float` である。
**明示するときは必ず `12.5`。**

---

## 3. parquet 化（`tools/prepare_dataset.py`）

```bash
for part in train dev; do
  uv run --python 3.12 --no-project \
    --with numpy --with pandas --with pyarrow --with tqdm \
    python -m tools.prepare_dataset \
      --tokenized_text_dir       "$D/$DS/$part/tok-text" \
      --tokenized_audio_dir      "$D/$DS/$part/tok-audio" \
      --output_prefix            "$D/$DS/parquet/$part" \
      --text_padding_id          3 \
      --num_examples_per_parquet 100000
done
```

- `--output_prefix` の **basename が `dialogue_id` の名前空間になる**（`train/v-001` 等）。
  `train` / `dev` 以外の名前にすると manifest と突き合わない。
- text と audio の stem 集合が違うと `StemMismatchError` で**落ちる**。黙って部分出力しない。
- `--text_padding_id` は tokenize_text と**同じ値**を渡す。ここは text stream を
  audio 長に合わせて詰める側の pad である。

---

## 4. 実行したコマンドを記録する（`tools/text_stream_audit.py record-tokenize`）

**tokenize_text を呼んだら必ずこれを呼ぶ。** M3 の欠陥は「フラグを落とした」ことではなく、
「どのフラグで走らせたか誰も記録しなかった」ことである。

```bash
uv run --python 3.12 --no-project python -m tools.text_stream_audit record-tokenize \
  --dataset_id   v-real-r1 \
  --manifest     experiments/tsukuyomi_ojousama/manifests/v-real-r1.jsonl \
  --out          experiments/tsukuyomi_ojousama/manifests/v-real-r1-tokenize.json \
  --split        train \
  --provenance   recorded \
  --recorded_at  2026-08-25 \
  -- \
  --word_transcript_dir       "$D/$DS/train/text" \
  --output_dir                "$D/$DS/train/tok-text" \
  --text_tokenizer_repo       nu-dialogue/j-moshi-ext \
  --text_tokenizer_name       tokenizer_spm_32k_3.model \
  --no_whitespace_before_word \
  --text_padding_id           3 \
  --end_of_text_padding_id    0 \
  --audio_tokenizer_frame_rate 12.5 \
  --num_workers               1
# dev は同じコマンドで --split dev と --append を付ける
```

`--` の後ろに `tokenize_text.py` の argv をそのまま貼る。argv は展開されて
`flags` / `defaults_used` に落ちる。**`--provenance recorded` を使えるのは、
実際に実行したコマンドをその場で貼ったときだけ**である（M3 の sidecar は
`reconstructed` になっている）。

`tests/test_experiment_assets.py::TokenizeFlagRecordTests` が
「全 dialogue manifest に sidecar があること」「sidecar が `no_whitespace_before_word` を
明示していること」「`false` なら `known_defect` を伴うこと」を CI で強制する。
**sidecar を書き忘れると CI が落ちる。**

---

## 5. 検算（`tools/text_stream_audit.py audit`）

```bash
uv run --python 3.12 --no-project --with numpy --with pandas --with pyarrow --with sentencepiece \
  python -m tools.text_stream_audit audit \
    "$D/$DS/parquet/train-001-of-001.parquet" \
    "$D/$DS/parquet/dev-001-of-001.parquet" \
    --text_tokenizer_path "$TOK" \
    --end_of_text_padding_id 0 \
    --out experiments/tsukuyomi_ojousama/reports/m3r-text-stream.json
```

見るのは `bare_whitespace_share_of_text_tokens`。**ゲートは 0.1。**
M3-R の想定値は **0.0**（本セッションで作り直した parquet の実測値）。

さらに sha256 を記録する。**parquet は `data/` 配下に置く**こと。
`tests/test_experiment_assets.py::TextStreamGateTests` は `data/` 以下を
`rglob("*.parquet")` で走査するので、`data/` の外に置いた parquet はゲートを素通りする。

```bash
shasum -a 256 "$D/$DS/parquet/"*.parquet
```

---

## 6. フラグ一覧 — 落とすと何が起きるか

| フラグ | 値 | 落とすと |
| --- | --- | --- |
| `--text_tokenizer_repo` | `nu-dialogue/j-moshi-ext` | **全対話が skip される**（英語 tokenizer で日本語が byte-fallback） |
| `--no_whitespace_before_word` | 立てる | text token の **44.6〜46.6% が裸の `▁`**。stream が 2.3 倍に伸びる。**下流は何も気づかない**（parquet は整合し、学習も始まる） |
| `--text_padding_id` | `3` | tokenizer を替えたときに pad が別 id になる |
| `--end_of_text_padding_id` | `0` | 同上 |
| `--audio_tokenizer_frame_rate` | `12.5` | `12` にすると text が音声から **2 秒ごとに 1 フレーム**ずれる。成果物に痕跡が出ない |
| `--num_workers` | `1` | 出力は変わらないが、ログの順序が崩れ skip の切り分けが難しくなる |
| `--device`（audio） | `cpu` | `cuda` は GPU 不在で `NoAcceleratorError`。`mps` は **bit 一致しない**（§1） |
| `--audio_chunk_size`（audio） | `1200` | 長い対話でメモリを踏む。M3-R の 60 秒超でも 1200 秒には遠い |
| `--output_prefix` の basename（parquet） | `train` / `dev` | `dialogue_id` の名前空間が変わり manifest と突き合わない |

---

## 7. 実行順のまとめ

```
1. audio  : tokenize_audio   --device cpu            → $D/$DS/{train,dev}/tok-audio
2. text   : tokenize_text    --no_whitespace_before_word → $D/$DS/{train,dev}/tok-text
3. parquet: prepare_dataset                          → $D/$DS/parquet/{train,dev}-001-of-001.parquet
4. 記録   : text_stream_audit record-tokenize        → manifests/<dataset_id>-tokenize.json
5. 検算   : text_stream_audit audit + shasum         → reports/m3r-text-stream.json
6. テスト : uv run --python 3.12 --with pytest --no-sync python -m pytest tests -q
```
