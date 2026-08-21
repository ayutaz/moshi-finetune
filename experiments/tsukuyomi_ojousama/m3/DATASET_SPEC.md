# V-real / V-tts データセット仕様と M3 合格基準

更新日: 2026-08-21

実行順序と費用は [M3実行計画](../../../docs/experiments/j-moshi-tsukuyomi-ojousama-m3-plan.md)、
完了判定は [マイルストーン文書](../../../docs/experiments/j-moshi-tsukuyomi-ojousama-milestones.md) を正とする。
この文書は **データセットの構造** と、**数値を見る前に固定した合格基準** を保持する。

`CLAUDE.md` は、根拠のない gate が動いている指標を3回却下した事例を記録している。
そのため本書の閾値は、候補が1つも存在しない時点で確定させる。データが出てから
決められる閾値は、データに合わせて決めた閾値と区別がつかない。

## 1. 二つのデータセット

V-real と V-tts は **チャンネルA のバイト列だけが違う**。それ以外はすべて共有する。

| 要素 | V-real | V-tts |
| --- | --- | --- |
| `dataset_id` | `v-real-v1` | `v-tts-v1` |
| 対話 script | 同一（`m3/scripts/dialogues-v1.jsonl`） | 同一 |
| 分割 | 同一（`m3/scripts/split-map-v1.json`） | 同一 |
| チャンネルA | つくよみちゃんコーパス **train 分割の原音** | M2採用 T1 speaker inversion による合成 |
| チャンネルB | 凍結話者B。**バイト単位で同一のwav** | 同左（同じファイルを参照） |
| 対話数 | 80（train 72 / dev 8） | 80（train 72 / dev 8） |
| ターン数 | 240（B開始・A・B終了 × 80） | 240 |

規模を揃えるのは、揃えないと V0 と V1 の差が「実音かTTSか」ではなく
**「データ量の差」**になり、比較が成立しないためである。

### 対話の形

1対話は3ターン。`B → A → B`。

- **A のターンはコーパスの1文をそのまま話す。** 文言を変更しない。
- B の前後発話はその文に自然につながるよう新規作成する。
- タイムラインは完全逐次。重なりなし、ターン間 **0.4 秒固定**、先頭 0.5 秒。
- チャンネル0 = A、チャンネル1 = B（`tools/tokenize_audio.py` の契約）。
- 24 kHz stereo。

barge-in を学習分布に入れない。turn-taking の**測定**はするが**学習はしない**。
これは限界であり、[M3実行計画の限界節](../../../docs/experiments/j-moshi-tsukuyomi-ojousama-m3-plan.md)に記載する。

### 分割規則

コーパスの dev / test 由来の文は **どちらのデータセットにも入らない**。
A の文集合は train 80文と集合として一致しなければならない。

V の train/dev 分割は **対話単位**で、記録した seed から割り当て、
`split-map-v1.json` に固定する。両データセットが同じ map を参照する。

継承ではなく制約として扱う理由: 全対話が train 由来なので「親の分割を継承」すると
すべて train になり、dev が作れない。したがって **(a) 親が dev/test の行は存在してはならない**
（テストで落とす）、**(b) V の分割は seed から割り当てる**、の2規則に分ける。

## 2. 話者B

VoiceDesign（`Aratako/Irodori-TTS-600M-v3-VoiceDesign`）の caption 条件で **1本だけ生成し、
凍結して以降すべての B 発話の `--ref-wav` に使う**。

発話ごとに caption から生成する方式は
[`m3-speaker-b-probe.json`](../reports/m3-speaker-b-probe.json) で否定された。

| 条件 | pairwise ECAPA 平均 | 最小 |
| --- | ---: | ---: |
| 実人間10本（校正基準） | 0.6983 | 0.5646 |
| caption毎回・seed既定 | 0.4525 | 0.4202 |
| caption毎回・seed固定 | 0.5176 | 0.4597 |
| **1本凍結して再利用** | **0.7373** | **0.7040** |

watermark は無効にする（`silentcipher` を import できない状態で生成する）。
学習データに透かしを入れる理由がない。

## 3. 事前登録した合格基準

**以下はすべて、対応する候補が存在しない時点で確定した。**

### 条件1: dataset 検証で不一致0

報告: `reports/m3-dataset-agreement.json`（ステップ12が書く）

160ペア（80対話 × 2データセット）すべてで、次の9つのカウントが **ちょうど0**。

| 検査 | 0でなければならない理由 |
| --- | --- |
| `channel_mismatches` | 左右が入れ替わると A が user stream になる。ここ以外で可視化されない |
| `timestamp_violations` | 語時刻が音声の外に出ると text stream が音声とずれる |
| `text_mismatches` | script と transcript の不一致 |
| `non_stereo` | `tokenize_audio` が assert で落ちる |
| `wrong_sample_rate` | — |
| `zero_length` | — |
| `below_min_frames` | **床は200 frames（16.0秒）**。下回る対話は script のバグであり、床を下げて通さない |
| `text_frames_exceeding_audio` | text stream が音声より長い |
| `saturated_files` | 連続飽和は実歪み（孤立サンプルは書き出し時の飽和で可聴でない。M2で確認済み） |

チャンネル判定には数値を伴わせる: A チャンネルの有声フレーム RMS が
B チャンネルのそれを **1.5倍以上**上回ること。左右入れ替わりはこの比が反転する。

### 条件2: tokenize の skip がすべて記録

報告: `reports/m3-tokenize-report.json`（ステップ14が書く）

- `verdict_count == 160` かつ `unaccounted_dialogues == 0`。
  全ペアが `accepted` か `rejected` を持ち、`rejected` はログからの逐語引用を伴う。
- データセットごとに `npz数 == wav数 == transcript数 == 80`。
- **skip は0件を要求する。** skip は吸収する損失ではなくデータのバグであり、
  ステップ11か12に戻って直す。

### 条件3: 独話 loop・反復 collapse なし

報告: `reports/m3-collapse.json`（ステップ32が書く）

少なくとも一方の run の採用 epoch で、固定30会話**すべて**について:

- `monologue_loop_count == 0`
- `exact_repeat_collapse_count == 0`

閾値は `reports/m3-collapse-calibration.json` に凍結する。**校正はステップ5で行い、
候補が存在する前に確定する。** 校正元は M0 の既払い生成20件。

> **校正の限界を明記する。** M0 の B チャンネルは無音であり、健全な control ではない。
> 校正は「崩壊の署名」に対して行っており、「健全さ」に対してではない。
> したがって `b_never_active` 系の指標を条件3の裏づけに使ってはならない —
> `--model_user_stream` 学習では user stream が全 arm で教師強制されるため、
> B は常に「活動中」に見え、この指標は**恒真になり判別力を持たない**。

### 条件4: held-out 話者らしさが改善

報告: `reports/m3-speaker-likeness.json`、`reports/m3-memorisation.json`（ステップ33・34）

held-out 10件の対応比較で、**次の3つをすべて満たす**。

| 基準 | 値 | 根拠 |
| --- | --- | --- |
| 方向 | `mean_delta > 0` | — |
| 一致 | 10件中 **8件以上**で control を上回る | — |
| 効果量 | `mean_delta >= +0.02` | M2で採用され聴取と整合した効果は `+0.03214` |

**加えて、暗記判定が `memorisation` でないこと。** V-real のチャンネルA は学習音声そのものであり、
`RUBRIC.md` は「seenだけが改善して held-out が改善しない checkpoint は暗記」と定める。

暗記判定は2つの入力を持つ。

1. **seen と held-out の差**: `seen_delta` が改善し `heldout_delta` が改善しない場合。
2. **逐語再生**: 生成テキストが学習文を **containment 0.8以上**で含む、または
   正規化部分文字列として完全一致するものが **1件でもあれば** `memorisation`。

containment を使い Jaccard を使わないのは、長い生成の中に短い学習文が丸ごと入った場合に
**Jaccard では見えない**ためである。実測（`tests/test_memorisation.py` が固定）:
50文字の学習文を124文字の生成が**逐語で丸ごと含む**とき、対称 Jaccard は **0.3967** で
既存の `_near_duplicate`（閾値0.9）は `False` を返す。containment は **1.0000**、
正規化部分文字列一致も `True`。union が生成側の追加分すべてを数えるため、
長さが違うだけで対称指標は希釈される。

**符号検定の `p` は参考値として記録し、判定には使わない。** held-out 10件では
9/1 で `p = 0.0215`、8/2 は `p = 0.1094` にしかならず、検出力が構造的に足りない。

### 条件5: 明瞭度と turn-taking が許容範囲

報告: `reports/m3-intelligibility.json`、`reports/m3-turn-taking.json`（ステップ35）

| 基準 | 値 |
| --- | --- |
| 盲検明瞭度（1〜5） | control 比 **0.5点以内**の低下（`RUBRIC.md`） |
| general 対話成功率 | control 比 **10ポイント以内**の低下（`RUBRIC.md`） |
| 応答開始遅延 | control 比 **2倍以内** |
| 話者交替数 | 30会話で **0にならない** |

`success_criteria` の絶対値（27/30）も測定して記録するが、**M3 を止めるのは相対規則の方**とする。
M3 は control との比較であって、絶対品質の到達点ではない。

> **明瞭度の限界を明記する。** 生成対話に正解テキストが無いため、CER は Moshi 自身が
> decode したテキストに対してしか測れない。自己参照であり、反復するモデルは低い CER を出しうる。
> 数値は補助であり、盲検聴取が判定を持つ。

### 条件6: 中間 checkpoint の採用理由

報告: `reports/m3-voice-control-gate.json`（ステップ37）

- 行数 == 10（2 run × 5 epoch）。
- 全10行に train loss と eval loss。
- 全評価 epoch に collapse 判定・対応 likeness delta・turn-taking・明瞭度。
- 全指標が揃わない epoch は、揃わない旨を明記する（欠測を空欄にしない）。

> **V0 と V1 の eval loss を同じ列に並べてはならない。** 各 run は自分の dev 音声で
> 評価しており、V-tts の dev 音声は TTS 由来である。比較可能なのは
> **run 内の epoch 間**だけである。

## 4. 生成物と、公開可否

| 種別 | 置き場所 | 公開 |
| --- | --- | --- |
| 対話 script（テキスト） | `m3/scripts/` | **する** |
| split map | `m3/scripts/` | **する** |
| manifest | `manifests/v-{real,tts}-v1.jsonl` | **する** |
| 検証・評価レポート | `reports/m3-*.json` | **する** |
| 生成 wav、原音、stereo 対話 | `data/.../m3/` | しない |
| tokenized npz、parquet | `data/.../m3/` | しない |
| checkpoint | `data/.../m3/` | 公開審査を通るまでしない |

原音とその派生音声は再配布しない。`DATA_CREDITS.md` を参照。

## 5. 中止条件

- VoiceDesign のライセンスが派生音声を制限していた場合、**M3 を止めて確認を取る**。
  話者Bは両データセットの全対話の片チャンネルを占めるため、権利判断であって技術判断ではない。
- 話者B候補が A との分離 0.30 を通らない場合、**bulk 生成に進まない**。
- 話者B変動の中央値が 0.65 を割った場合、**tokenize せずに話者Bを作り直す**。
- 対話が200フレームの床を割った場合、**B のターンを長くする。床は下げない**。
