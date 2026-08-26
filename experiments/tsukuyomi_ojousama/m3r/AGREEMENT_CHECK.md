# M3-R 一致検査の手順 — 9 種の検査を出荷 dataset に当てる

作成日: 2026-08-27
対象: `v-real-v2`
道具: [`tools/dataset_agreement.py`](../../../tools/dataset_agreement.py)
出力: [`reports/m3r-dataset-agreement.json`](../reports/m3r-dataset-agreement.json)
比較対象: [`reports/m3-dataset-agreement.json`](../reports/m3-dataset-agreement.json)

M3 は 9 種の一致検査を行い、その結果だけを記録した。**検査を行ったスクリプトは
コミットされていない。** 同じことが第2段のビルドでも起きており
（[`DATASET_BUILD.md` §0](./DATASET_BUILD.md)）、この文書と `tools/dataset_agreement.py` は
その再発を止めるためにある。数値は手で書き写さない。`report` 部分コマンドが
測定ファイルから組み立てる。

---

## 1. 前提

```bash
cd /Users/inamotoyuuta/Desktop/moshi-finetune
NICE="nice -n 19"                                  # このマシンの負荷を上げない
ENV2="OMP_NUM_THREADS=2 MKL_NUM_THREADS=2"
OUT=$(mktemp -d)                                   # 測定の中間ファイル置き場
DEPS='--with numpy --with pandas --with pyarrow --with soundfile
      --with pyopenjtalk-plus --with "torch==2.4.1" --with "torchaudio==2.4.1"'
```

`pyopenjtalk-plus` は読み比較に、`torchaudio` は相槌 wav の 48 kHz → 24 kHz 変換に使う。
変換器は assembler と同じもの（`torchaudio.functional.resample`）を使う。**変換器は
検査対象ではないので、別のものを使うと 2 つの変換器の差を測ることになる。**

GPU は使わない。実測 78 対話で約 40 秒である。

---

## 2. 測る（出荷そのまま）

```bash
$ENV2 $NICE uv run --python 3.12 --no-project $DEPS python -m tools.dataset_agreement check \
  --dataset_root data/experiments/tsukuyomi_ojousama/m3r/v-real \
  --manifest     experiments/tsukuyomi_ojousama/manifests/v-real-v2.jsonl \
  --dialogues    experiments/tsukuyomi_ojousama/m3r/scripts/dialogues-v2.jsonl \
  --split_map    experiments/tsukuyomi_ojousama/m3r/scripts/split-map-v2.json \
  --dataset_id   v-real-v2 \
  --excluded     v-047 v-057 \
  --out          "$OUT/agree.json"
```

対話は **manifest の行から** 引く。ディレクトリを glob しない。glob は出荷されていない
材料（`m3r/v-real/audio/` の 80 本）を拾う。

## 3. 測る（チャンネルを入れ替えて）

**この段は省略できない。** 落ちない検査は検査ではない。

```bash
$ENV2 $NICE uv run --python 3.12 --no-project $DEPS python -m tools.dataset_agreement check \
  ...同じ引数... --negative_control --out "$OUT/agree-swapped.json"
```

`channel_mismatches` が 0 のままなら、検査が入れ替えを見ていないということであり、
出荷側の 0 も証拠にならない。実測では 752 件・78 対話すべてが落ち、相槌 NCC は
1.0000 から中央値 0.184 に落ちた。

## 4. parquet を直接測る

`check` は tok-audio npz からフレーム数を取る。parquet 自体の行数・checksum・
裸 `U+2581` は別に取り、報告に並べる。**2 つが食い違えば、どちらかの台帳が古い。**

```bash
$ENV2 $NICE uv run --python 3.12 --no-project --with numpy --with pandas --with pyarrow \
  python - > "$OUT/parquet.json" <<'PY'
import hashlib, json, statistics
from pathlib import Path
import numpy as np, pandas as pd
base = Path("data/experiments/tsukuyomi_ojousama/m3r/v-real/parquet")
out = {}
for split in ("train", "dev"):
    path = base / f"{split}-001-of-001.parquet"
    frame = pd.read_parquet(path)
    frames = [np.asarray(cell.tolist()).shape[1] for cell in frame["A"]]
    bare = total = 0
    for _, row in frame.iterrows():
        for speaker in ("A", "B"):
            merged = np.asarray(row[speaker].tolist())
            bare += int((merged[0] == 9).sum())
            total += int(merged.shape[1])
    out[split] = dict(
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        byte_size=path.stat().st_size,
        rows=len(frame),
        frames=dict(min=min(frames), max=max(frames), median=statistics.median(frames),
                    mean=round(statistics.fmean(frames), 3), total=sum(frames)),
        bare_whitespace_pieces=bare, text_frames=total,
        streams_per_speaker_column=sorted({np.asarray(c.tolist()).shape[0] for c in frame["A"]}),
    )
print(json.dumps(out, indent=1))
PY
```

## 5. 報告を組み立てる

```bash
$NICE uv run --no-sync python -m tools.dataset_agreement report \
  --measured         "$OUT/agree.json" \
  --negative_control "$OUT/agree-swapped.json" \
  --parquet          "$OUT/parquet.json" \
  --annotations      experiments/tsukuyomi_ojousama/m3r/agreement-annotations.json \
  --out              experiments/tsukuyomi_ojousama/reports/m3r-dataset-agreement.json
```

`--annotations` は**文章だけ**を持つ。数値は測定ファイルから来る。文章が数値を
上書きできる構造にはしてあるが、上書きするのは方法の説明であって観測ではない。

終了コードは、9 種すべてが 0 かつ M3-R 固有の追加検査もすべて通ったときだけ 0 である。

---

## 6. M3 の文言から変えたもの（2 件）

変更は**閾値の緩和ではなく、測る対象の訂正**である。M3 自身が同じ訂正を 2 回している
（チャンネル全体 RMS 比 → ターンごと、表層一致 → 読み一致）。

| 検査 | M3 の文言 | なぜ通らないか | M3-R |
| --- | --- | --- | --- |
| `channel_mismatches` | 話者のチャンネルがエネルギーを持ち、もう一方が持たない | もう一方はルームトーンを持ち、相槌は A のターンに重なる。どちらも設計 | 排他フレームでの**比較**（(a)）＋ 同フレームでの speech 閾値（(b)） |
| `text_frames_exceeding_audio` | text stream が音声より長い | `tokenize_text.py` が末尾に 1 秒（12.5 フレーム）padding を必ず付ける。字義どおりなら M3 の dataset も全件落ちる | audio のフレーム数を超えた位置にある**padding 以外の**トークン数 |

**(a) を絶対量にしてはならない。** 最初は「中央値 RMS >= 0.01」と書いた。話者 A の
154 ターン中 22 件が落ち、その 22 件は同じ対話のルームトーン（中央値 0.00021）の
30〜40 倍上にあった。この閾値が見つけたのは無音ではなく、**話者 A が話者 B より
19.16 dB 静かである**という dataset 全体の性質である。左右の入れ替えは絶対量を
変えないので、絶対量には検出力がない。

---

## 7. 結果（2026-08-27）

9 種すべて 0。M3-R 固有の追加検査は 2 件通過・1 件不合格。

| | 実測 |
| --- | ---: |
| 対話 | 78（train 70 / dev 8） |
| ターン | 386（うちチャンネル判定可能 377） |
| 9 種の不一致 | すべて 0 |
| 判定余裕（出荷） | 最小 +9.56 dB / 中央値 +53.26 dB |
| 判定余裕（入れ替え） | 最大 −9.56 dB / 中央値 −53.26 dB |
| 2 群の分離 | 19.11 dB、間に 1 件もない |
| 相槌 NCC | 76 本すべて 1.0000（最小 0.99999999723） |
| 相槌の位置ずれ | 最大 0.07 秒（許容 1 フレーム = 0.08 秒） |
| ルームトーン | 156 チャンネル全部、厳密ゼロの連続 0.0 秒 |
| 除外 2 対話 | parquet・tok-*・manifest から消えている。**split-map には残っている** |

**不合格 1 件と、その隣で見つかったもう 1 件**は
`reports/m3r-dataset-agreement.json` の `mismatches_found` に記録した。どちらも
このステップでは直していない。**検査の役目は報告であって、データを直すことではない。**

1. `split-map-v2.json` が v-047 / v-057 を train として持ったままで、`counts` も 72/8 のまま
2. `reports/m3r-tokenize.json` が出荷前の 72 行 parquet を記録したまま
   （checksum・byte_size・frames.min・`launch_assertions.Num examples` すべて）

2 は金がかかる。第4段 4-2 のゲートは「起動 assertion が `Num examples 72`」であり、
出荷 parquet では trainer が **70** と印字する。**正しい run が古い台帳を根拠に
kill される。** 正しい値は同報告の `correct_launch_assertion` にある。
