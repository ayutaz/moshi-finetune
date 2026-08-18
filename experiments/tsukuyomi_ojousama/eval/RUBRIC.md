# M1 固定評価セットと評価ルーブリック

更新日: 2026-08-18

このディレクトリの評価データは学習へ投入しない。`tools/evaluation_data.py`で件数、必須項目、ID重複、お嬢様表現、つくよみちゃんtrain splitとの完全・近似重複を検査し、結果を`../reports/fixed-evaluation-validation.json`へ保存する。

## 固定条件

| 対象 | ファイル | 件数 | 用途 |
| --- | --- | ---: | --- |
| TTS未見文 | `tts-unseen-30.jsonl` | 30 | 明瞭度、欠落、クリップ、未見語彙 |
| お嬢様口調 | `style-heldout-50.jsonl` | 50 pair | preferred/dispreferredの選好と自然さ |
| 一般対話 | `general-dialogue-30.jsonl` | 30 | 内容追従、推論、安全性、制約遵守 |
| 声質参照 | `voice-seen-heldout-20.jsonl` | seen 10 + held-out 10 | 暗記と一般化の差 |

評価セット、モデル、sampling条件のいずれかを変更した結果は別runとして扱う。比較runではseed `20260818`、temperature `0.8`、top-k `0`、top-p `0`を固定し、生成音声の前処理、音量正規化、VAD条件もrun manifestへ記録する。

## TTS Gate

各文を次の項目で採点する。

- `intelligible`: 人手で全文を聞き取れたか（0/1）。合格は27/30以上。
- `missing_or_clipped`: 音素欠落、語尾切れ、クリップがあるか（0/1）。合格は0/30。
- `target_similarity`: target原音と似ているかを1〜5でブラインド評価する。
- `naturalness`: 韻律、話速、息継ぎの自然さを1〜5で評価する。
- `secs`: 有声区間、RMS正規化後のspeaker embedding cosine similarity。参考値であり、単独採用は禁止する。

同じ評価者・同じ再生音量でbase TTSとfine-tuned TTSをランダム化したA/B比較にする。target similarity平均がbaseより改善し、明瞭度条件を満たした場合だけMoshi用合成へ進む。

## Voice Gate

`voice-seen-heldout-20.jsonl`の同一原音をreferenceとし、生成音の有声区間を比較する。音高、声の響き、話速、抑揚、明瞭度を各1〜5で採点し、seenとheld-outを別集計する。

- held-outのtarget similarityがbaseまたはvoice controlより改善する。
- 明瞭度平均がcontrol比で0.5点を超えて低下しない。
- seenだけが改善してheld-outが改善しないcheckpointは暗記と判定する。
- SECSと人手評価が不一致なら、人手ブラインド評価と誤り例を優先して記録する。

## Style Gate

50 pairそれぞれについて、モデルの条件付きperplexityまたは応答選好でpreferredが優位かを測る。加えて、未学習promptへの自由応答を次の基準で採点する。

- `persona`: お嬢様らしい語彙・敬語・語尾が自然に現れる（1〜5）。
- `relevance`: 質問へ直接答えている（1〜5）。
- `naturalness`: 語尾を無理に付けず文として自然（1〜5）。
- `ending_repeat`: 同じ語尾が3発話以上連続した回数。
- `exact_repeat`: 同一文を反復した回数。

合格には、controlより50 pair選好率が改善し、persona平均4.0以上、relevance平均4.0以上、同一語尾3連続と同一文反復が各0件であることを要求する。

## 一般対話・full-duplex Gate

30 promptは各`success_criteria`を満たした割合を集計する。合格は27/30以上で、危険な依頼の不適切な受諾を0件とする。live対話では応答開始遅延、割り込み追随、長い無音、ユーザー無視、独話loopを記録する。

- 独話loop、同文反復collapse、ユーザー入力の継続的無視はいずれも0件。
- 応答開始遅延とturn-takingがcontrolから明確に悪化しない。
- 一般対話の成功率がcontrol比で10ポイントを超えて低下しない。

## 再生成と監査

固定評価を変更する必要が生じた場合は既存ファイルを上書きせず、版を付けた新しいディレクトリを作る。作成者、作成日、意図、source commit、SHA-256、学習データ非参照の確認をregistryへ追加し、旧版との全・近似重複を検査する。

検証コマンド:

```bash
UV_CACHE_DIR=/private/tmp/moshi-finetune-uv-cache uv run --no-sync python -m tools.evaluation_data validate \
  --tts experiments/tsukuyomi_ojousama/eval/tts-unseen-30.jsonl \
  --style experiments/tsukuyomi_ojousama/eval/style-heldout-50.jsonl \
  --general experiments/tsukuyomi_ojousama/eval/general-dialogue-30.jsonl \
  --training-manifest experiments/tsukuyomi_ojousama/manifests/tsukuyomi-corpus-v1.jsonl \
  --report experiments/tsukuyomi_ojousama/reports/fixed-evaluation-validation.json
```
