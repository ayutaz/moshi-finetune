> **これは手順書ではない。** `TOKENIZE_COMMANDS.md` を**丸ごと置き換えるための本文**である。
> 適用方法と、なぜ直接書き換えられなかったかは
> [`PENDING_CORRECTIONS.md`](./PENDING_CORRECTIONS.md) を読むこと。
> 適用したらこのファイルを削除する。

---

# M3-R tokenize コマンド — 全フラグ、実行順、落としたときに何が起きるか

作成日: 2026-08-25
更新日: 2026-08-26（出荷物の `dataset_id` に名前を合わせ、audio と parquet のフラグも記録対象にした）
確定した工程: [M3-R 計画](../../../docs/experiments/j-moshi-tsukuyomi-ojousama-m3r-plan.md) 第 2 段 2-1
実証記録: [`reports/m3r-tokenize-fix.json`](../reports/m3r-tokenize-fix.json)
前後の工程: [`DATASET_BUILD.md`](./DATASET_BUILD.md)（split 材料化から manifest・検証まで）

この文書は **M3-R の dataset を作るときにそのまま実行するコマンド**を持つ。
すべて本セッションで実行し、出力を確認したものである。推定で書いた行はない。

M3 はコマンドラインをどこにも記録しなかった。`m3/tokenize.log` に残っていたのは
`tokenize_text rc=0` の 4 行だけで、`--no_whitespace_before_word` が落ちていたことは
**出荷から 3 日後に parquet を数え直して初めて分かった**。この文書はその再発を止めるためにある。

出荷する dataset は **`v-real-v2`** である（`m3r/scripts/split-map-v2.json`、
`manifests/v-real-v2.jsonl`）。以前この文書の §4 は `v-real-r1` と書いていた。
手順どおり実行すると出荷物と別の `dataset_id` ができるので、直した。

---

## 0. 前提

| 項目 | 値 | 理由 |
| --- | --- | --- |
| Python | 3.12 | プロジェクトの `.venv` は 3.13 で torch が入らない |
| 実行形式 | `uv run --no-project` | `uv.lock` は gitignore。依存はコマンドに書き切る |
| cwd | リポジトリ root | `python -m tools.*` が `tools` を見つけるため |
| dataset_id | `v-real-v2` | 出荷物。`v-real-v1` は M3 の（欠陥のある）dataset |
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
ID=v-real-v2                                       # 出荷する dataset_id
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

> **この値は §4 で manifest の sidecar に記録する。** `tools/tokenize_flags.py` の
> `check_record` は、`device` が `mps` の記録に `known_defect` が付いていなければ
> 問題として返す。`tests/test_tokenize_flags.py` がその negative control を持つ。

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

`--text_padding_id` は §3 の `prepare_dataset` にも渡す。**同じ値でなければならない。**
食い違うと 1 本の stream の中で「無音」を綴る id が 2 つになり、parquet はどちらでも整合する。
`tools/tokenize_flags.py` の `check_record` がこの食い違いを検出する。

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
- `--tokenized_text_dir` / `--tokenized_audio_dir` は **§1 と §2 が書いた先**でなければならない。
  前回の tokenize の出力を読んだ parquet は、行数も列も checksum も自分自身とは整合する。
  **自分の入力とだけ整合しない。** これが M3 の失敗の一段上の形である。

---

## 4. 実行したコマンドを記録する（`tools/text_stream_audit.py record-tokenize`）

**tokenize を呼んだら必ずこれを呼ぶ。** M3 の欠陥は「フラグを落とした」ことではなく、
「どのフラグで走らせたか誰も記録しなかった」ことである。

記録先は manifest の隣の sidecar `manifests/<dataset_id>-tokenize.json` である。

### 4-1. なぜ 3 回呼ぶか — 選んだ方式とその理由

`record-tokenize` は当初 `tokenize_text.py` の argv しか記録しなかった。
`tokenize_audio` と `prepare_dataset` のフラグは `reports/m3r-tokenize.json` にしか残らず、
**manifest からは辿れなかった**。落ちて一番痛いのは `--device cpu` である（§1）。

**既存の仕組みでは賄えない。** sidecar のスキーマは 1 つの invocation が
`tokenize_text` の全フラグを持つことを CI が強制しており（`TokenizeFlagRecordTests`）、
別のツールの invocation を同じ配列に足すと、そのテストが「フラグ表が不完全」として落ちる。
記録できないのは記法の問題ではなく、**検証が 1 ツールを前提に書かれていた**ためである。

**したがって仕組みを変えた。** フラグ表を `tools/tokenize_flags.py` に集め、3 ツール分を持たせ、
invocation ごとに `tool` を書くようにした。`record-tokenize` に `--tool` が付いた（既定は
`tokenize_text` なので、既存の呼び出しはそのまま動く。`tool` を持たない過去の invocation も
`tokenize_text` として読む）。

代替案として「reports/ を manifest から参照させる」も検討したが採らなかった。
参照先が別ファイルなら、参照が古くなっても誰も落ちない。**フラグは manifest の隣に置く。**

### 4-2. コマンド

```bash
# 1) audio。--device がここに落ちる
uv run --python 3.12 --no-project python -m tools.text_stream_audit record-tokenize \
  --tool         tokenize_audio \
  --dataset_id   "$ID" \
  --manifest     "experiments/tsukuyomi_ojousama/manifests/$ID.jsonl" \
  --out          "experiments/tsukuyomi_ojousama/manifests/$ID-tokenize.json" \
  --split        train \
  --provenance   recorded \
  --recorded_at  2026-08-26 \
  -- \
  --audio_dir              "$D/$DS/train/audio" \
  --output_dir             "$D/$DS/train/tok-audio" \
  --audio_tokenizer_repo   kyutai/moshiko-pytorch-bf16 \
  --audio_tokenizer_name   tokenizer-e351c8d8-checkpoint125.safetensors \
  --audio_chunk_size       1200 \
  --device                 cpu \
  --num_workers            1

# 2) text。以降はすべて --append を付ける（付け忘れると前の記録が消える）
uv run --python 3.12 --no-project python -m tools.text_stream_audit record-tokenize \
  --tool         tokenize_text \
  --dataset_id   "$ID" \
  --manifest     "experiments/tsukuyomi_ojousama/manifests/$ID.jsonl" \
  --out          "experiments/tsukuyomi_ojousama/manifests/$ID-tokenize.json" \
  --split        train \
  --provenance   recorded \
  --recorded_at  2026-08-26 \
  --append \
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

# 3) parquet
uv run --python 3.12 --no-project python -m tools.text_stream_audit record-tokenize \
  --tool         prepare_dataset \
  --dataset_id   "$ID" \
  --manifest     "experiments/tsukuyomi_ojousama/manifests/$ID.jsonl" \
  --out          "experiments/tsukuyomi_ojousama/manifests/$ID-tokenize.json" \
  --split        train \
  --provenance   recorded \
  --recorded_at  2026-08-26 \
  --append \
  -- \
  --tokenized_text_dir       "$D/$DS/train/tok-text" \
  --tokenized_audio_dir      "$D/$DS/train/tok-audio" \
  --output_prefix            "$D/$DS/parquet/train" \
  --text_padding_id          3 \
  --num_examples_per_parquet 100000
```

**`dev` について同じ 3 本を、`--split dev --append` で繰り返す。**
合計 6 回。1 回目だけ `--append` を付けない。

`--` の後ろにそのツールの argv をそのまま貼る。argv は展開されて
`flags` / `defaults_used` に落ちる。**`--provenance recorded` を使えるのは、
実際に実行したコマンドをその場で貼ったときだけ**である（M3 の sidecar は
`reconstructed` になっている）。

### 4-3. 何が検証されるか

`tools/tokenize_flags.py` の `check_record` が、記録どうしの矛盾を返す。
いずれも「起きたことのある失敗」であって、思いついた規則ではない。

| 検査 | 落ちると何が起きるか |
| --- | --- |
| ツールの全フラグが値付きで存在する | `store_true` を落とした run と記録しなかった run が区別できない（M3） |
| `device` が `mps` なら `known_defect` を伴う | 同じ wav から別の parquet。下流は何も気づかない |
| `text_padding_id` が text と parquet で一致 | 1 本の stream に「無音」の id が 2 つ |
| `prepare_dataset` の入力が §1・§2 の出力と一致 | 前回の tokenize から作った parquet。自分自身とだけ整合する |
| `--output_prefix` の basename が split 名 | `dialogue_id` の名前空間が変わり manifest と突き合わない |

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

stream 数（trainer が受け取る 17）と held-out の混入は
[`DATASET_BUILD.md`](./DATASET_BUILD.md) §5 が持つ。

---

## 6. フラグ一覧 — 落とすと何が起きるか

| ツール | フラグ | 値 | 落とすと |
| --- | --- | --- | --- |
| text | `--text_tokenizer_repo` | `nu-dialogue/j-moshi-ext` | **全対話が skip される**（英語 tokenizer で日本語が byte-fallback） |
| text | `--no_whitespace_before_word` | 立てる | text token の **44.6〜46.6% が裸の `▁`**。stream が 2.3 倍に伸びる。**下流は何も気づかない**（parquet は整合し、学習も始まる） |
| text | `--text_padding_id` | `3` | tokenizer を替えたときに pad が別 id になる。parquet 側と食い違うと 1 stream に無音が 2 種類 |
| text | `--end_of_text_padding_id` | `0` | 同上 |
| text | `--audio_tokenizer_frame_rate` | `12.5` | `12` にすると text が音声から **2 秒ごとに 1 フレーム**ずれる。成果物に痕跡が出ない |
| text | `--num_workers` | `1` | 出力は変わらないが、ログの順序が崩れ skip の切り分けが難しくなる |
| audio | `--device` | `cpu` | `cuda` は GPU 不在で `NoAcceleratorError`。`mps` は **bit 一致しない**（§1）。**既定は `cuda`** なので、名指ししないとローカルでは走らない |
| audio | `--audio_chunk_size` | `1200` | 長い対話でメモリを踏む。M3-R の 1 行は 19〜24 秒で、1200 **秒**には遠い |
| audio | `--num_workers` | `1` | 出力は不変。M3 と揃える |
| parquet | `--output_prefix` の basename | `train` / `dev` | `dialogue_id` の名前空間が変わり manifest と突き合わない |
| parquet | `--tokenized_{text,audio}_dir` | §1・§2 の出力 | 前回の tokenize から作った parquet ができる。自分自身とだけ整合する |

> **`--audio_chunk_size` の注記について。** この行は以前「M3-R の 60 秒超でも」と書いていた。
> 系列長 60 秒という目標は
> [M3-R データセット監査](../../../docs/experiments/j-moshi-tsukuyomi-ojousama-m3r-dataset-audit.md) §2.5
> で**撤回された**（このプロジェクトで唯一動いた run は 19.02 秒だった）。
> 出荷する 1 行は 19〜24 秒である。

---

## 7. 実行順のまとめ

```
1. audio  : tokenize_audio   --device cpu                → $D/$DS/{train,dev}/tok-audio
2. text   : tokenize_text    --no_whitespace_before_word → $D/$DS/{train,dev}/tok-text
3. parquet: prepare_dataset                              → $D/$DS/parquet/{train,dev}-001-of-001.parquet
4. 記録   : text_stream_audit record-tokenize --tool ... → manifests/v-real-v2-tokenize.json
            3 ツール × 2 split = 6 回。1 回目以外は --append
5. 検算   : text_stream_audit audit + shasum             → reports/m3r-text-stream.json
6. テスト : uv run --python 3.12 --with pytest --no-sync python -m pytest tests -q
```

**4 を飛ばすと 1〜3 が消える。** それが M3 で起きたことである。
