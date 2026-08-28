# M3-R 実行計画 — 計器とデータを直してから、V-real で control を取り直す

作成日: 2026-08-24
更新日: 2026-08-28 — **4-2 run1 を実行し、失敗した。**学習に到達せず US$4.289 を消費した。
その事実と、そこから決まる予算・手順を書き直した。
（2026-08-27 — 第0〜3段と 4-1 が完了したので、予測で書いた箇所を実測で置き換えた。）

前提文書:
- [M3 実施報告](./j-moshi-tsukuyomi-ojousama-m3-report.md)（**原因診断は撤回済み**）
- [M3 検証記録](./j-moshi-tsukuyomi-ojousama-m3-verification.md)（撤回の根拠と、確定した 12 件）
- [M3-R データセット監査](./j-moshi-tsukuyomi-ojousama-m3r-dataset-audit.md)（系列長ゲートの撤回、acoustic 損失重み）
- [マイルストーン](./j-moshi-tsukuyomi-ojousama-milestones.md)（完了判定の正本）
- [run1 失敗の記録](../../experiments/tsukuyomi_ojousama/reports/m3r-run1-failure.json)
  （4-2 の停止位置、否定した仮説、課金 US$4.289）
- [第4段の打ち切り線](../../experiments/tsukuyomi_ojousama/m3r/STOP_LINE.md)
  ／ 正本の数値 [`reports/m3r-stop-line.json`](../../experiments/tsukuyomi_ojousama/reports/m3r-stop-line.json)
- [V-real データセット仕様と合格基準](../../experiments/tsukuyomi_ojousama/m3/DATASET_SPEC.md)（条件1〜6 の正本）

**この文書の読み方。** 各段は「計画」と「実績」を並べて持つ。計画の文言は消していない。
**達成したもの・撤回したもの・置き換わったもの**を区別できる形にするためである。
実測値には出典を付けた。**この文書と出典が食い違えば出典が正**である。

---

## 進捗（2026-08-28 現在）

| 段 | 内容 | 状態 | 課金 | 主な証拠 |
| --- | --- | --- | ---: | --- |
| 第0段 | 記録を直す | **完了** | US$0 | commit `0c00024` |
| 第1段 | 計器を直す | **完了** | US$0 | [`m3-collapse-acoustic.json`](../../experiments/tsukuyomi_ojousama/reports/m3-collapse-acoustic.json)、[`m3-likeness-calibration.json`](../../experiments/tsukuyomi_ojousama/reports/m3-likeness-calibration.json)、[`m3-text-stream-audit.json`](../../experiments/tsukuyomi_ojousama/reports/m3-text-stream-audit.json) |
| 第2段 | データを作り直す | **完了**（`v-real-v2` 出荷） | US$0 | [`m3r-tokenize-fix.json`](../../experiments/tsukuyomi_ojousama/reports/m3r-tokenize-fix.json)、[`m3r-script-validation.json`](../../experiments/tsukuyomi_ojousama/reports/m3r-script-validation.json)、[`m3r-roomtone.json`](../../experiments/tsukuyomi_ojousama/reports/m3r-roomtone.json)、[`m3r-timeline.json`](../../experiments/tsukuyomi_ojousama/reports/m3r-timeline.json)、[`m3r-tokenize.json`](../../experiments/tsukuyomi_ojousama/reports/m3r-tokenize.json) |
| 第3段 | ローカルで測り切る | **完了** | US$0 | [`m3r-roundtrip.json`](../../experiments/tsukuyomi_ojousama/reports/m3r-roundtrip.json)、[`m3r-intelligibility.json`](../../experiments/tsukuyomi_ojousama/reports/m3r-intelligibility.json)、[`m3r-dataset-agreement.json`](../../experiments/tsukuyomi_ojousama/reports/m3r-dataset-agreement.json)、[`m3r-stop-line.json`](../../experiments/tsukuyomi_ojousama/reports/m3r-stop-line.json) |
| 第4段 4-1 | base loss の内訳を forward で測る | **完了** | **US$0.115** | [`m3r-forward-breakdown.json`](../../experiments/tsukuyomi_ojousama/reports/m3r-forward-breakdown.json) |
| 第4段 4-2 | V-real 学習と変換（run1） | **実行・失敗（成果ゼロ）** | **US$4.289** | [`m3r-run1-failure.json`](../../experiments/tsukuyomi_ojousama/reports/m3r-run1-failure.json) |
| 第4段 4-3 以降 | export・生成 | **未着手**（run1 が通るまで着手できない） | 下の第4段を参照 | — |

**4-2 は「未着手」ではない。走らせて、失敗した。**学習に到達せず、起動 assertion にも届いていない。
詳細は下の第4段「4-2 run1: 実行と失敗」節にある。

累計 `accrued_estimate` **US$107.301**（[`m0/spend-ledger.json`](../../experiments/tsukuyomi_ojousama/m0/spend-ledger.json)）。
このうち第4段で使ったのは US$4.604 で、**成果が残っているのは 4-1 の US$0.115 だけである。**
インスタンスは 3 つとも破棄済みで、**この実験の日次課金は US$0.00 である。**

**現行の上限では run1 を再挑戦できない。** preflight は `reject-new-run` を返す（下の予算節）。

---

## この計画が答える問い

M3 は不合格だった。判定は正しいが、**原因の読みは誤りだった**。
したがって M3-R は「M3 のやり直し」ではない。**M3 が答えられなかった問いに、
初めて答えられる状態を作る**ことが目的である。

答えるべき問いは 1 つ。

> **データ構造と学習率を文献・過去成功例に合わせたとき、J-Moshi-ext は
> つくよみちゃんの声に寄りながら対話能力を保てるか。**

M3 はこの問いを立てられていなかった。データが無音を教え、計器が音を見ず、
基準線が壊れていたためである。**第0〜3段でその 3 つはすべて片付いた。**

---

## 承認済みの決定（2026-08-24）と、その後の実績

| 項目 | 決定 | 実績（2026-08-28） |
| --- | --- | --- |
| GPU 上限 | **US$100 → US$125**（+US$25） | 変更なし。累計 **107.301**、上限までの残余 US$17.699。ただし拘束するのは下の preflight 限度である |
| 再走の腕 | **V-real のみ** | 変更なし。V-tts の教師は較正帯の外（0.7213、35/80） |
| 寄せ集めコーパス | **申請しない** | 申請していない |
| forward 測定の見積もり | US$0.40 | **実績 US$0.115**（0.382 h × US$0.3017/h、V100 32GB 1 枚）。差額 US$0.285 は残余に戻った |
| V-real 再走の見積もり | US$18 | **1 回実行し、失敗した。**US$4.289 を使って学習に到達していない。再挑戦にはさらに run1 1 本ぶんが要り、**現行上限では preflight を通らない**。第4段を参照 |

> **`cap_raise` の「US$18.40 が買える」という記述は、preflight の観点では成立しない。**
> `tools/experiment_budget.py` が判定するのは上限そのものではなく **上限の 90%（US$112.50）** である。
> 累計 **107.301** に対し preflight が許す残余は **US$5.199** しかない
> （第3段 3-4 の時点では 102.812 に対し US$9.688 だった。失敗した run1 がその差を食った）。
> 判定の枠組みは [`STOP_LINE.md` §0](../../experiments/tsukuyomi_ojousama/m3r/STOP_LINE.md) が確定させたが、
> **同文書の数値は 102.812 時点のものである**（下の「記録の突き合わせ」）。

---

## 課題の全体像

M3 の失敗は 1 つではなく、**独立した 5 群**である。第 4 段（GPU）に到達する前に、
第 0〜3 段（すべて課金ゼロ）で 4 群を潰す —— **その 4 群は潰し終えた。**

| 群 | 内容 | 直す段 | 実績 |
| --- | --- | --- | --- |
| **A. 記録** | 撤回された原因診断が 4 文書 6 箇所に残存 | 第 0 段 | **解消。** 残るのは訂正バナーと本計画のゲート文自身のみ |
| **B. 計器** | 基準線 control が 16/30 退化、崩壊検出器が音を見ない、条件 4 の検出力 0.383、eval loss 不在、seed 未記録 | 第 1 段 | **解消。** 音響指標つき検出器、条件4 の区間推定、eval loss 内訳（4-1 が実地で使った）、seed 必須化 |
| **C. データ** | text の 45.5% が裸の `▁`、A が 1 回しか話さない、A 無音 68.8%、重なりゼロ | 第 2 段 | **解消。** 裸 `▁` 0.0000%、A のターン 1.975、重なり 1,202 frame、デジタル無音 0/13,919。ただし `pad ⟺ 無音` の**一致率という指標**は 60% 未満に届いていない（2-4） |
| **D. 手法** | `3e-5` は事前学習の LR、warmup 0、全パラメータ更新 | 第 4 段の設定に反映 | LR は下表で確定。warmup は事前登録済みの 0（実効 2 step）で決着。**acoustic 損失重みのみ未決**（第4段） |
| **E. 予算** | 打ち切り線を守れなかった 3 つの原因 | 第 4 段の運用に反映 | 打ち切り線は UTC 時刻で確定済み。4-1 は 1.50 h の線に対し 0.382 h（25%）で終わった。**run1 も 3.376 h の線に対し 1.72 h で自主中断しており、線は守れている。**守っても成果ゼロだったのは、線とは別の理由である（下の 4-2） |

---

## 第 0 段: 記録を直す（US$0 / 約 1 時間）— **完了**

**blocking。** これを済ませずに次へ進むと、以後のすべての設計が撤回済みの前提の上に乗る。

| # | 作業 | ゲート | 実績 |
| --- | --- | --- | --- |
| 0-1 | `m3-report.md` を[検証記録 §9](./j-moshi-tsukuyomi-ojousama-m3-verification.md) の 10 件で訂正 | 10 件すべてに対応する差分がある | **通過**（commit `0c00024`、`m3-report.md` に 257 行の差分） |
| 0-2 | 「4.6%」「1/24」を 4 文書 6 箇所で訂正 | `grep -rn "4\.6%\|1/24" docs/` が検証記録以外 0 件 | **実質通過。** 残存は (a) 検証記録、(b) `m3-report.md` 冒頭の**訂正バナー自身**、(c) 本行のゲート文。誤診として残っている箇所は 0 件 |
| 0-3 | `milestones.md` の M3 記述を「原因未特定」に更新し、M4 の Blocked 理由を差し替え | M4 の Blocked 理由が「control checkpoint 不在」＋「計器 3 件」になっている | **通過。** マイルストーン文書の M4 完了記録がその 2 点を名指ししている |

**訂正の要点**: 「データ量が主因」→ **「原因未特定。データ構造・学習率・更新範囲の 3 つを
M3 は同時に振っており、分離されていない」**

---

## 第 1 段: 計器を直す（US$0 / 約 3 時間）— **完了**

**blocking。** 計器が壊れたまま再走しても、**再走の結果も同じ理由で判定できない。**

| # | 作業 | ゲート | 実績 |
| --- | --- | --- | --- |
| 1-1 | `tools/dialogue_collapse.py` に音響指標を追加（話者 A codebook0 の distinct / 最頻占有率 / entropy）。閾値は calibration ファイルに固定 | control の general30 が **16/30 退化**と自動判定される | **通過（値は 17/30）。** 較正済み規則（`entropy<=1.43 OR distinct<=9`）で **17/30**。探索的規則（`distinct<=8 AND top>=0.7`）は検証記録の **16/30** を厳密に再現する。較正済み規則は探索的規則が挙げるものを常に含む |
| 1-2 | 既存 550 生成を新指標で再判定し、`m3-collapse.json` を上書きせず**別ファイル**に出す | 全 11 腕 × 3 prompt set が再判定済み | **通過。** [`m3-collapse-acoustic.json`](../../experiments/tsukuyomi_ojousama/reports/m3-collapse-acoustic.json) に 11 腕 × 3 set。550 件中 21 件が「text は健全・音響は崩壊」、105 件がその逆で、2 つは同じものの別の見方ではない |
| 1-3 | `tools/speaker_similarity.py` の出力に較正帯・絶対 cosine・per-clip σ を必須化 | 出力 JSON に 3 項目すべてが存在する | **通過。** `require_likeness_report` が 3 項目を欠いた報告を raise する |
| 1-4 | 条件 4 の判定を「8/10 符号検定」→「paired mean + 区間推定 + 較正帯までの距離の何%を閉じたか」へ置換。`m3/DATASET_SPEC.md` を改訂 | 新基準が**候補を見る前に**文書化されている | **通過。** 2026-08-25、M3-R の候補 checkpoint が 0 件の時点で凍結。較正帯 mean 0.8166 / floor 0.7405、control 0.3728、距離 0.4438、バーは距離の **25%**（= 0.4838） |
| 1-5 | 記録間不一致（v-tts/epoch3 の scorable 9 対 10）の原因を特定し解消 | 差の原因が特定され、記録されている | **通過。** 原因は degeneracy guard。`likeness_guard.apply_degeneracy_guard` が clip 4 を `exact_repeat_collapse` として候補側から除いていた。再測定は guard を通らない経路だった（`m3-likeness-calibration.json` の `record_reconciliation`） |
| 1-6 | `finetune.py` を修正: gathered 平均を印字、loss 内訳 4 項目を `--with_tracking` なしで stdout へ、eval loss の出力を smoke test で確認 | テストが通り、内訳 4 項目が stdout に出る | **通過。** `gather_metric_means` / `format_loss_breakdown` / `format_evaluation_log_line`。**4-1 はこの出力から内訳 4 項目を取った** —— 直した計器が実地で効いた最初の例である |
| 1-7 | `generate.py` の seed を必須化し、生成ごとに記録 | seed なしの生成が実行できない | **通過。** argparse で `--seed` を required にし、config 書き出し時にも seed 不在を raise する |

**1-1 の閾値についての注記**: 音響側の閾値は 2026-08-25 に凍結しており、
**候補（M3 の 550 生成）が既に存在した後**である。calibration ファイルはそれを明記したうえで、
判定が妥当な閾値域で平坦であることを示している。**候補 0 件の時点で凍結した 1-4 とは事情が違う。**

**計画に無かったが第1段で直った 2 件**（どちらも検証の副産物である）

| 何 | なぜ重要か |
| --- | --- |
| `DEFAULT_MIN_BAND_CLOSURE` を 0.25 に pin し、mutation test を付けた | 0.05 に動かしても 473 件のテストが全部通っていた。**腕の合否を決める定数がテストに守られていなかった** |
| `tools/experiment_budget.py` の `HARD_CAP = 100.0` を上限からの比率へ修正 | 上限が 125 に上がった後も 100 のままで、**あらゆる計画に非ゼロを返していた**。「止まれ」と言える唯一の計器が、何も意味しない状態だった。第3段・第4段の preflight はすべて修正後の道具で取っている |

---

## 第 2 段: データを作り直す（US$0 / 半日〜1 日）— **完了、`v-real-v2` を出荷**

**この段が最大の勝負どころである。** 検証記録が確定させた最有力の機序は
「**loss を下げる最短経路が無音だった**」であり、それを作ったのはデータ構造である。

### 目標と実測

| 項目 | M3 | 過去成功 run | M3-R の目標 | **実測（出荷 `v-real-v2`）** | 出典 |
| --- | ---: | ---: | --- | ---: | --- |
| turn / 対話 | 3.00 | 6.12 | 5.0 以上 | **4.95** | `m3r-script-validation.json` |
| A が話す回数 / 対話 | **1.00** | 約 3.06 | 2 以上 | **1.975** | 同上 |
| 1 ターンの平均長 | 7.0 秒 | 3.1 秒 | 短くする | 分割後の断片は平均 19.9 文字（最短 7・最長 38） | 同上 |
| A が無音の時間 | 68.8% | — | 下げる | **66.5%**（clip 基準・train）。**この指標はほとんど動いていない** | `m3r-timeline.json` |
| ~~系列長~~ | ~~21.0 秒~~ | 19.02 秒 | **撤回**（[監査](./j-moshi-tsukuyomi-ojousama-m3r-dataset-audit.md) §2） | train 中央値 **244 frame = 19.52 秒**（201〜302 frame） | `m3r-dataset-agreement.json` |
| 重なり | なし | — | 入れる | **1,202 frame**（train 1,084 / dev 118）、**重なりの無い対話 0 件** | `m3r-timeline.json` |
| 非発話チャンネル | デジタル無音 | — | ルームトーン | **デジタル無音 0/13,919 フレーム**（M3 は 13,535/15,548 = 87.05%） | `m3r-roundtrip.json` |
| text の裸 `▁` | 45.5% | — | 3.6%（フラグ付き） | **0.0000%**（train / dev、A / B とも 0 個） | `m3r-tokenize.json` |
| train step 数 | 45 | 65 | 45 を維持 | **45**（`ceil(70/8) × 5`） | 同上 |
| train / dev 行数 | 72 / 8 | — | 72 / 8 | **70 / 8**（200 frame の床を割った v-047・v-057 を除外） | `m3r-timeline.json` |

**目標より良かったもの**: 裸 `▁` は 3.6% ではなく **0.0000%** だった。目標の 3.6% は
「フラグを付けても残る」という推定で、実測では 1 個も残らない。

**目標に届かなかったもの**: turn/対話 4.95（目標 5.0）、A のターン 1.975（目標 2）。
読点を持たない 2 文（`v-011`「クィーンズアベニューアルファに所属している。」、
`v-042`「ウォリアーズミックスマーシャルアーツアカデミー所属。」）が分割できず、
この 2 対話だけ B-A-B のままである。**80 対話中 78 が B-A₁-b-A₂-B になった。**

**動かなかったもの**: 「A が無音の時間」は 68.8% → 66.5% でほとんど変わっていない。
A の発話量が増えていないのだから当然で、**この指標を下げること自体は第2段の目的ではなかった。**
目的は `pad ⟺ 無音` の近道を切ることであり、それはルームトーンが達成した（下の 2-4）。

### 設計上の制約と、その解き方

train 文は 72 文しかない。**A のターン数を増やすと対話数が減り、step が落ちる。**

| A が話す回数 | train 対話 | 総 step | 系列長 |
| ---: | ---: | ---: | ---: |
| 1（M3 現行） | 72 | 45 | 21 秒 |
| 2（素朴に分割） | 36 | **25** | 46 秒 |
| 3 | 24 | **15** | 70 秒 |

**解: 文を読点で分割する。** train 72 文のうち **70 文（97%）が「、」を含み、63 文が 2 個以上**持つ。
1 文を 2 ターンに割れば、**対話数を 72 のまま**、A のターンを 2 回にできる。

**この解は機能した**（実測 78/80 対話、turn/対話 4.95、A turn/対話 1.975）。
なお上の表の「系列長」列は連結を前提に書かれていたが、連結は撤回した。
分割だけなら対話数は 72 のまま、系列長は約 19 秒、step は 45 である。

```
M3  :  B ──── A ──────── B                      3 turn / 21 秒
M3-R:  B ── A₁ ─ b ─ A₂ ──── B                  4.95 turn / 平均 244.6 frame = 19.6 秒
              ↑ 短い相槌（「ええ」「なるほど」）。A の休止に重ねる
```

さらに以下を重ねた。

1. **ルームトーン** — 非発話チャンネルにコーパス収録の実際の無音区間を敷く。
   `pad ⟺ 無音` の近道を直接切る。**最も直接的で最も安い。**
2. **重なり** — A の発話終端と B の相槌を数百 ms 重ねる。full-duplex を教える。
3. ~~**連結**~~ — **撤回した。**「Kyutai 推奨 100〜300 秒」は README の誤読（`300` は一度も現れず、`duration_sec` は上限の指定）であり、過去の成功 run は 19.02 秒・1 対話 1 例・`ceil(100/8)×5 = 65 step` だった。連結は唯一動いた前例から遠ざかる。[監査](./j-moshi-tsukuyomi-ojousama-m3r-dataset-audit.md) §2 を参照。

### 手順と実績

| # | 作業 | ゲート | 実績 |
| --- | --- | --- | --- |
| 2-1 | `--no_whitespace_before_word` を付けて既存 parquet を作り直し、行単位で diff | row 1-16 が**バイト単位で不変**。text の裸 `▁` が 45.5% → 3.6% | **通過。** audio 行 **1,280 行すべてが不変**。裸 `▁` は 44.64〜46.64% → **0.00%**、text token は 12,154 → 4,754（−60.9%）。フラグ無しで作り直した parquet は出荷済みと sha256 まで一致し、差が「フラグのせい」だと分離できている |
| 2-2 | 読点分割の台本を生成。分割点の自然さを全 72 文で目視確認 | 分割後の各ターンが 2 モーラ以上、文として読める | **通過。** 80 文中 78 文を分割、断片は最短 7 文字。**A の総文字数 3,149 が分割前後で不変** —— 台本を書き換えたのではなく形を変えたことの検算である |
| 2-3 | 相槌ターンの台本を作り、話者 B を追加生成（VoiceDesign 固定 ref-wav、ローカル MPS） | 話者 B 一貫性が M3 と同水準（平均 0.8390、最悪 0.6542） | **ゲートの文言どおりの測定は行っていない。** 新しい相槌 wav に対する ECAPA 一貫性はリポジトリのどの報告にも無い。代わりに置かれた証拠は (a) M3 と**バイト単位で同じ**凍結 ref-wav、(b) seed 20260825 と 12 種の定型相槌、(c) 出荷チャンネルと相槌 wav の最良ラグ NCC が全数 1.0000（組み上げ 78 本／出荷 76 本） |
| 2-4 | ルームトーンを抽出（コーパス原音の無音区間）し、非発話チャンネルに敷く | `pad ⟺ A 無音` の一致率が 98% → **60% 未満** | **ゲートの数値としては未達。** 転写区間基準で 0.907（M3 出荷）→ 0.809（作り直し）までしか下がらない（[マイルストーン](./j-moshi-tsukuyomi-ojousama-milestones.md)の完了条件に記録）。一致率は定義で大きく動き、同一 parquet で 0.9070 / 0.9911 / 0.8508 になる。**このゲートが代理していた「loss を下げる最短経路が無音」の方は、pad 率ではなく token 側で断った**: 話者 A の静音フレームの**デジタル無音率 0/13,919**（M3 は 87.05%）、無音テクスチャ {1316, 2029} の占有率 71.4% → 15.8% |
| 2-5 | 重なりを入れてタイムラインを再構成 | 重なり区間が全対話に存在し、両チャンネル同時発話のフレームが 0 でない | **通過。** 同時発話 1,202 frame（うち音響的に両方鳴っているもの 663）、**重なりの無い対話 0 件**（M3 は 80/80 が重なり無し） |
| ~~2-6~~ | **撤回。** 連結しない | — | 系列長は中央値 19.52 秒（244 frame）で出荷。train 70 / batch 8 / 5 epoch = **step 45** で M3 とも過去の成功 run とも構成が揃う |
| 2-7 | 新 dataset を tokenize し、manifest と registry を更新 | `tests/test_experiment_assets.py` が通る。step 数が **45** | **通過。** `v-real-v2` を cpu で tokenize（mps は bit-identical でない）。train 70 行 / dev 8 行、streams 17、sha256 `adea0474…` / `f21d821e…`、step 45 |
| 2-8 | tokenize の全フラグを manifest に記録し、text 統計をテストのゲートにする | フラグを落とすと CI が落ちる | **通過。負の対照で確認済み。** フラグ無しの parquet を `data/` 配下に置くと `TextStreamGateTests` が `0.4464 not less than 0.1` で落ちた。落ちることを見ていないゲートは主張であって測定ではない |

**2-1 が最重要のゲート**だった。row 1-16 が不変だったので、
「audio 側の崩壊軌跡（distinct 81 → 5）は text 欠陥では説明できない」が確定した。
**崩壊の原因は別に探す必要がある**という含意も同時に確定している。

### 床を割った 2 対話（計画に無かった判断）

組み上げた 72 行のうち 2 行（`v-047` `v-057`、いずれも 198 frame）が
`m3/DATASET_SPEC.md` の 200 frame の床を割った。**床は下げず、2 行を出荷から外した。**
規則が指定する対処（B のターンを伸ばす）は、相槌が A の休止に重なる構造では対話長を伸ばさない。
別の 3 行（`v-026` `v-039` `v-079`）は **lead-in を 0.3 から M3 と同じ 0.5 に戻した**ことで床を越えた
（0.3 になっていた理由は記録がなく、床を通すために選んだ値ではない）。
`ceil(70/8) = 9`、`9 × 5 = 45` で、**総 step は 72 行のときと同じ 45 である。**

---

## 第 3 段: ローカルで測り切る（US$0 / 数時間）— **完了**

GPU を借りる前に、**借りずに測れるものを全部測る。**

| # | 作業 | ゲート | 実績 |
| --- | --- | --- | --- |
| 3-1 | 学習投入 token を Mimi で decode して WAV に戻し、統計と耳で確認 | A が明瞭なつくよみちゃん、B が中性話者として鳴る | **統計は全項目通過。耳では聴いていない**（`m3r-roundtrip.json` の `limits[0]` が明記）。チャンネル入れ替わり無し（包絡相関 10/10、最小マージン 1.22）、delay の二重適用無し（20 チャンネルすべて lag 0）、decoded-A は**コーデック調整帯**の平均 −0.0102 に位置し 10 件中 9 件が floor 以上 |
| 3-2 | 明瞭度を測る。held-out と seen の decode 済み 110 件を Whisper にかけ、転写の日本語 LM perplexity を取る。反復検出と併記 | 条件 5 が初めて数値化される。control の値が基準になる | **通過。** control（held-out + seen 20 件）は **clean_transcribed_ratio 0.80**、median perplexity 2523.1（反復込み）／3285.2（反復除外）。M3 報告書の空欄が埋まった |
| 3-3 | 新 dataset に対し M3 と同じ 9 種の一致検査を実行 | 不一致 0 | **9 種すべて 0。** 負の対照（チャンネル入れ替え）で 78/78 が落ちるので、0 は検出力の裏付きである。**M3-R 固有の追加検査で 1 件不合格**（下記） |
| 3-4 | 打ち切り線を**UTC 時刻**で確定し、export 時間を見積もりに含める | 時刻が文書に書かれている | **通過。** 線の引き方・ステップ別締切・走行中の手順を [`STOP_LINE.md`](../../experiments/tsukuyomi_ojousama/m3r/STOP_LINE.md) に固定。同時に「**現行の計画のままでは第4段を開始できない**」という判定が出た |

**3-2 の注意**: 反復するモデルは perplexity が下がりうる。**必ず反復検出と併記する。**
これは実測で裏付いた。M3 で最も崩壊した 2 腕（v-real/epoch4・epoch5）の median perplexity は
**197.8** で 33 群中ほぼ最良、実在の人間の録音（916.5）より 4.6 倍「流暢」である。
崩壊を捉えるのは **`clean_transcribed_ratio`**（分母固定、無音と反復の両方を課金する量）であり、
それでも epoch に対して単調ではない。**条件 5 は `clean_transcribed_ratio`・`repetitive_transcripts`・
`empty_transcript` の 3 つを並べて読む。**

**3-3 が見つけた不合格 1 件（第4段に直結する）**

`reports/m3r-tokenize.json` が出荷前の 72 行 parquet を記録したままだった。
**この計画の 4-2 のゲートは「起動 assertion が `Num examples 72`」と書いていた。**
出荷 parquet では trainer は **70** と印字する。
つまり**正しい run が、古い台帳を根拠に kill される**ところだった。
行数・sha256・byte_size は修正済みで、正しい起動 assertion は下の 4-2 に書き直してある。

---

## 第 4 段: GPU 再走 — 4-1 完了、**4-2 は 1 回失敗**、以降が残り

**予算の現在地**（[`m0/spend-ledger.json`](../../experiments/tsukuyomi_ojousama/m0/spend-ledger.json)）

| | |
| --- | ---: |
| 累計 `accrued_estimate` | **US$107.301** |
| 上限 | US$125.00 |
| 上限までの残余 | US$17.699 |
| **preflight の限度（`new_run_prediction_limit`）** | **US$112.50** |
| **preflight が許す残余** | **US$5.199** |
| 走行中の停止線 | US$118.75（残余 US$11.449） |

**拘束するのは US$17.699 ではなく US$5.199 である。**
`tools/experiment_budget.py` は `spent + rate × hours` を上限の 90% と比べる。
**この計画の旧見出し「US$25 以内」も、「上限までの残余 US$22.30」という読み方も、この限度を見ていなかった。**

**第4段がここまでに使った額**（台帳の `charges` と `accrued_estimate` の内訳）

| instance | 何 | 実績 | 課金 | 残ったもの |
| ---: | --- | --- | ---: | --- |
| 48838452 | 4-1 forward 測定 | 0.382 h × US$0.3017/h | **US$0.115** | loss 内訳 4 項目（[`m3r-forward-breakdown.json`](../../experiments/tsukuyomi_ojousama/reports/m3r-forward-breakdown.json)） |
| 48911444 | 4-2 の 1 回目の選定 | 走らせずに破棄 | **US$0.20** | 「`dph_total` は disk 代を含まない」という教訓 |
| 48911872 | 4-2 run1 | 1.72 h × US$2.4936/h | **US$4.289** | **無し。**学習に到達していない |
| | | | **US$4.604** | うち成果があるのは **US$0.115** |

**48911444 を走らせずに破棄した判断は正しかった。** 検索が示した US$2.0896/h に対し実請求は US$3.3327/h で、
差は `storage_cost` US$1.00/GB/月 × 900 GB = **+US$1.23/h** である。その率では budget_line が 2.91 h となり、
work_line 3.376 h を下回る —— 変換の途中で打ち切ることになる。**借りる前に率を計算し直すこと。**

```
rate = dph_total + storage_cost × disk_gb / 730
```

### 設定

| 項目 | M3 | **M3-R** | 根拠・実績 |
| --- | --- | --- | --- |
| 腕 | V-real + V-tts | **V-real のみ** | V-tts の教師が較正帯の外 |
| dataset | `v-real-v1` | **`v-real-v2`** | train 70 / dev 8、sha256 `adea0474…` / `f21d821e…`。**4-1 で instance 側と手元の sha256 一致を確認済み** |
| tempformer LR | 3e-5 | **2e-6** | J-Moshi 自身の fine-tuning 値 |
| depformer LR | 3e-5 | **4e-6** | 同上 |
| warmup | 0（明示せず既定） | **0 を明示**（実効 2 step） | [実験計画](./j-moshi-tsukuyomi-ojousama-plan.md) が候補 0 件の時点で事前登録した値。DeepSpeed が `max(2, ·)` にクランプするので実効 2 step で、M3 と同一。リファレンスの 500 は総 step 45 の 11 倍で最終 step でも目標 LR の 9%。下の「未決 2 → 決着」 |
| `--max_length` / `--min_length` | 省略 | **2048 / 128 を明示** | 195〜302 frame の行では両方 no-op。明示は省略の検出のためである（[監査](./j-moshi-tsukuyomi-ojousama-m3r-dataset-audit.md) §4） |
| acoustic 損失重み | 1（既定） | **未決** | 4-1 が「A 側の実効寄与 1.3%」を確定させた。下の「未決 1」 |
| 更新範囲 | 全パラメータ | 全パラメータ（`--parameters_to_finetune` は実装済みで、次の変数として保留）| 一度に振る変数を減らす |
| export | epoch 3・5 のみ | **全 epoch** | M3 は最良の epoch 2 を失った |
| seed | 未指定 | **固定** | 全計数が 1 回の抽選だった。`generate.py` は seed 無しでは動かない |
| 生成サンプル数 | prompt あたり 1 | **複数** | 単発抽選では分散が測れない |

### 4-1: base loss の内訳を forward で測る — **完了（2026-08-27、US$0.115）**

M3 で 2 arm × 5 epoch = 10 回計算されながら `--with_tracking` の内側でしか出ず、
**全部捨てられた 4 項目**を初めて取得した。dev 8 行、LR = 0、重みは動いていない。

| 項目 | loss | audio_total に占める割合 | accuracy | chance 比 |
| --- | ---: | ---: | ---: | ---: |
| `audio_semantic`（話者A） | 2.57052 | **15.7%** | 37.6% | 33.7% |
| `audio_acoustic`（話者A） | 2.97663 | **1.3%** | 30.0% | 39.0% |
| `audio_semantic_user`（複製ヘッド） | 13.14448 | **80.1%** | **1.6%** | **172.4%** |
| `audio_acoustic_user`（複製ヘッド） | 7.03738 | 3.0% | 6.7% | 92.3% |

- 重み付き和が報告値 `audio_total = 7.67102` と**小数第5位まで一致**する。読みは推論ではなく算術である
- text 側: `text_non_pad` 5.18883、`text_pad` 0.34235、`text_total` 5.36000、`loss/total` 13.03102
- chance = `ln(2048)` = 7.6246

**この測定が確定させたこと**

1. **M3 の base audio loss 6.82〜7.19 は「j-moshi-ext が対象話者を予測できない」ことの証拠ではなかった。**
   その 83.1% は `models/utils.py` が deepcopy で作った**未学習の複製ヘッド**が占めている。
   A 側だけ見れば semantic 2.57（chance の 33.7%）、accuracy 37.6% で、モデルはよく予測できている。
2. **acoustic の実効重みは A 側 1.3%** である。名目 3.27% より小さいのは、複製ヘッドが押しのけるためである。
3. 複製ヘッドは chance を大きく超える（172.4%）。**自信を持って外している**という、未学習ヘッドの挙動である。

### 4-1 で判明した運用上の事実（4 件）

| 事実 | 何が起きたか | 4-2 以降への影響 |
| --- | --- | --- |
| **`finetune.py` は DeepSpeed 必須** | [`finetune.py:425-426`](../../finetune.py) が `NotImplementedError: Only DeepSpeed is supported for now.` を投げ、1 回目の起動が死んだ | 単一 GPU でも `accelerate launch --use_deepspeed --deepspeed_config_file …` で起動する。optimizer offload が不要でも config は M3 と揃える |
| **`m3/bootstrap_m3_instance.sh` は 2× A100 80GB を要求して停止する** | GPU 2 枚未満・80,000 MiB 未満・利用可能 RAM 80 GiB 未満のいずれかで `FATAL` を出して exit 1 する | **gate は緩めない。**学習には正しい（ZeRO-3 + AdamW で 133.94 GB 常駐）。forward には過剰なので、**forward 用の別 gate（20 GiB カード / 40 GiB RAM）**を置いた。ログに `forward gate passed: 32768 MiB card, 720 GiB RAM` が残っている |
| **forward 用 gate はコミットされていない** | 実体は instance 上のスクリプトで、記録は `data/…/forward-out/bootstrap-forward.log`（gitignore 配下）にしかない | 4-2 以降で 2 種類目の箱を借りるなら、**gate をリポジトリに置いてから借りる** |
| **eval は決定的である** | `eval_dataloader` は `shuffle=False`（[`finetune.py:886`](../../finetune.py)）で dev 8 行を全件通す。LR = 0 なら重みも動かない | seed 未指定でも 4-1 の値は再現する。**学習 run では seed を固定する**（設定表） |

**dq16 base model の構築は決定的**である。`--init_text_embeddings` を付けなければ乱数は入らず、
**16,742,873,520 bytes** になる。run1 の base model 構築はこのバイト数で検算できる。

### 4-2 run1: 実行と失敗（2026-08-27、US$4.289、成果ゼロ）

正本は [`reports/m3r-run1-failure.json`](../../experiments/tsukuyomi_ojousama/reports/m3r-run1-failure.json) と
[`m0/spend-ledger.json`](../../experiments/tsukuyomi_ojousama/m0/spend-ledger.json) の instance 48911872 の項である。
**この文書と食い違えばそちらが正である。**

**学習に到達していない。**下の 4-2 手順でいえば、段 d の起動 assertion
（`Num examples 70` / `Total train batch size 8` / `Total optimization steps 45`）に**一度も届いていない**。
したがって loss も checkpoint も存在せず、**この run から判定に使える材料は 1 つも出ていない。**

#### 停止位置

| | |
| --- | --- |
| ログ上 | `Generating train split: 70 examples` の直後。次に出るはずの `Loading eval dataset from ...` が出ない |
| コード上 | [`finetune.py:820-827`](../../finetune.py) の `with accelerator.main_process_first():` を抜ける地点。`Loading eval dataset from ...` は同 830 行で、そこへ進んでいない |
| 手元に残る診断 | `data/experiments/tsukuyomi_ojousama/m3r/run1-diag/`（`train.log` / `run1.nohup` / `procinfo.txt` / `nvidia-smi.txt`。**gitignore 配下**）。これは `--dataset_processing_workers 1` で再起動したぶんで、ログの最終行は 17:05:53 の `Loading train dataset from [...]`。21 分後（17:26:52）の `nvidia-smi` でも先へ進んでいない |

#### 症状

- 両ランクが **CPU 100%**。utime は増加し続ける（10 秒で 10 秒分）
- **45 秒間、HF datasets キャッシュにも `/workspace` にも書き込みが無い**
- `nvidia-smi.txt`: 2 枚とも **GPU-Util 100%、VRAM 597 MiB、71W / 300W**。
  モデルは載っていないのに GPU が回り続け、電力は上がらない
- NCCL は init まで到達している（17:04:52 の `Initializing TorchBackend in DeepSpeed with backend nccl`）
- `VmRSS` は両ランクとも 33,649,872 kB（約 32.1 GiB）。fp32 の CPU ロード（1 ランク 31.19 GiB）は終わっている

**「CPU も GPU も 100%、I/O だけがゼロ」は、前処理が遅いときの形ではない。**
spin-wait の形である。

#### 否定した仮説 3 件 —— **うち 2 件は私の誤診である**

| 仮説 | 検証 | 結果 |
| --- | --- | --- |
| `num_proc=16` の並列 map が原因 | `--dataset_processing_workers 1` で再起動（**課金あり**） | **同じ地点で 19 分停止。否定。**並列度は関係ない |
| `futex_wait_queue` だからデッドロック | 実処理ランクの utime を 10 秒間隔で比較 | **増加していた。**見ていたのは**親プロセスの** wchan だった。**私の誤診。**デッドロックではなく busy-wait である |
| `preprocess_function` が重い（`split_streams` / `filter_out_short_streams` は `--max_length` / `--min_length` を明示した M3-R の新経路で、M3 は通っていない） | **出荷 parquet 70 行を手元で通した（課金ゼロ）** | **0.01 秒で完了。**出力 70 例・shape (17,280)。フラグの有無で差なし。**否定。**私の誤診であり、**1.25 時間の停止（課金あり）のあいだ疑っていた経路が、手元では 0.01 秒だった** |

**この 3 件目が、この run のいちばん高い教訓である。**
CPU だけで完結する段を、GPU の時間を買ってから疑った。
`CLAUDE.md` の「Measure the premise before building on it」は、この形でもう一度破られている。

#### 残った仮説 — **NCCL の P2P が A100 PCIe で成立していない**

| | GPU | 出典 | 結果 |
| --- | --- | --- | --- |
| M3（成功） | **A100-SXM4-80GB**（NVLink） | [`m3-instance-bootstrap.json`](../../experiments/tsukuyomi_ojousama/reports/m3-instance-bootstrap.json) | 45 step 完走 |
| M3-R run1（失敗） | **A100 80GB PCIe** | `run1-diag/nvidia-smi.txt` | collective でハング |

`vastai search offers` の表示はどちらも "A100" である。**両者を区別せずに借りたのは、私の選定ミスである。**
control の再現に、測るつもりのなかった変数（相互接続）を持ち込んだ。

**状態は「未検証」である。** GPU を借りないと確かめられない。
CPU 100% + GPU-Util 100% + I/O ゼロは NCCL の busy-wait と整合するが、**整合は証明ではない。**
`m3r-run1-failure.json` の `why_m3_did_not_hit_it` が置く別の可能性
（M3 は train 側をキャッシュから読んでおり、この load 経路の所要が違った）も、まだ消えていない。

**安い検査方法がある。** 次に借りるとき `NCCL_DEBUG=INFO` を付け、`--max_train_steps 2` で 2 step だけ回す。
通らなければ `NCCL_P2P_DISABLE=1` を足してもう一度。**数分で判る。**手順は下の 4-2 段 c に組み込んだ。

**単一 GPU への切り替えは逃げ道にならない。** ZeRO-3 + fp16 + AdamW は 16 bytes/param = 133.94 GB 常駐で、
1 枚 85.90 GB に対し 48 GB 足りない（`.claude/skills/vast-run/SKILL.md` の表）。
`--parameters_to_finetune` を絞れば載る可能性はあるが、**未検証であり、control の変数を 1 つ増やす。**

#### なぜ 1.72 h で止めたか

| | |
| --- | ---: |
| 打ち切り線（work_line） | 3.376 h |
| 実績 | **1.72 h**（自主中断） |
| うち停止していた時間 | 約 1.25 h（56 分 + 19 分） |

中断時点の残りは 1.78 h。そこに残っていた仕事は**学習 1.07 h + 変換 0.5 h = 1.57 h** で、
猶予は **0.21 h** しかなかった。単一 GPU への切り替えを試して失敗すれば、変換に届かない。
**変換されない ZeRO state は、破棄した瞬間にゼロになる。**
線を守って原因を持ち帰る方を選んだ —— この判断自体は正しい。**正しい判断でも、成果はゼロである。**

---

### 4-2 以降をどう分けるか — **案 B の分け方は維持。予算の結論は失効した**

**「現行 US$125 の内側。追加承認は要らない」は、run1 が 1 回失敗した今、成立しない。**
US$4.289 は戻らず、accrued は 102.812 → **107.301** に上がった。
**分け方（run1 = 学習と変換 / run2 = 安い箱で export）そのものは変わらない。**変わったのは値段である。

第3段の積み上げでは、単箱で全部やると **7.041 時間 / US$21.52** である。
そのうち **export の 2.718 時間（US$8.31）は 84 GB を手元へ落とす転送**で、
律速は手元の回線（実測 8.56 MB/s）であって、インスタンスではない。
**US$3.0567/時の機械でダウンロードを待っている。**

| 案 | 内容 | 必要上限（accrued 107.301 で引き直し） | 判定 |
| --- | --- | ---: | --- |
| A | 単箱で 7.041 時間 | **US$143.14**（= 128.823 ÷ 0.90） | **通らない。**`--planned-hours 7.041 --hourly-rate 3.0567` は exit 1（予測 128.823） |
| **B** | **run1 = 学習と変換、run2 = 安い箱で export** | **US$127.76 〜 US$131.25**（下表） | **現行 US$125 では通らない。**上限の判断が要る |
| C | export する epoch を 5 → 3 | 1.087 時間 US$3.32 の節約 | **採らない。**全 epoch export は M3 が最良の epoch 2 を失ったことへの対策そのものである。しかも US$3.32 では足りない |
| D | 第4段をここで止める | — | 第0〜3段と 4-1 の成果は残るが、control checkpoint が得られず M4 は Blocked のまま |
| **S** | **診断だけ買う**（SXM4 を借り、段 a〜c の smoke test で止める） | **現行 US$125 のままで通る** | 0.9 h で予測 **109.545**、1.5 h で **111.041**、どちらも exit 0。**checkpoint は得られない。**得られるのは「SXM4 なら collective を抜けるか」という答えだけである |

**案 S を選ぶと、そのぶん本番の残余が減る。** 1.5 h 買えば accrued は 111.041 になり、
preflight の残余は US$1.459 —— **上限を上げない限り、その後に run1 は走らない。**
上限を上げるつもりがあるなら、smoke test は独立の run ではなく **run1 の段 c** として買う方が安い。

**「安い箱は本当にあるのか」は実測で確認済み（2026-08-27）。**
RTX 3060 12GB が **US$0.0525/h**、GTX 1070 が **US$0.0481/h** で実在する。
第3段が仮定していた US$0.45/h より 1 桁安い。
**ただしこれは `dph_total` であって、disk 代を含まない**（48911444 の教訓）。
`rate = dph_total + storage_cost × disk_gb / 730` を借りる直前に計算し直すこと。
（[`STOP_LINE.md` §6 B](../../experiments/tsukuyomi_ojousama/m3r/STOP_LINE.md) と
[`m3r-stop-line.json`](../../experiments/tsukuyomi_ojousama/reports/m3r-stop-line.json) は
US$0.45/h の仮定と accrued 102.812 のままである。**台帳の `offer_surveys` には記録済み**で、そちらが正である。）

**preflight の実測**（すべて `uv run --no-sync python -m tools.experiment_budget --spent 107.301`、
2026-08-28 に実行。**定数は緩めていない**）

| 対象 | `--hourly-rate` | `--planned-hours` | 予測 | status | exit |
| --- | ---: | ---: | ---: | --- | ---: |
| **run1 再挑戦（見積もり 2.876 h、run1 の実請求率）** | 2.4936 | 2.876 | **114.473** | **reject-new-run** | **1** |
| run1 再挑戦（同、M3 の SXM4 率） | 3.0566666666666666 | 2.876 | 116.092 | reject-new-run | 1 |
| run1 再挑戦（予備 0.5 h こみ = work_line） | 2.4936 | 3.376 | **115.719** | reject-new-run | 1 |
| 同上、SXM4 率 | 3.0566666666666666 | 3.376 | 117.620 | reject-new-run | 1 |
| いま通る限界（US$2.4936/h） | 2.4936 | 2.084 / **2.085** | 112.498 / **112.500** | allow / **reject** | 0 / **1** |
| いま通る限界（US$3.0567/h） | 3.0566666666666666 | 1.70 / **1.71** | 112.497 / **112.528** | allow / **reject** | 0 / **1** |
| 案 S（診断だけ） | 2.4936 | 0.9 / 1.5 | 109.545 / 111.041 | allow-with-warning | 0 |
| 案 A（単箱） | 3.0566666666666666 | 7.041 | 128.823 | reject-new-run | 1 |

> 3 行目の 115.719 は上のコマンドの出力である。
> [`m3r-run1-failure.json`](../../experiments/tsukuyomi_ojousama/reports/m3r-run1-failure.json) は
> 115.718 と記録している（時間の丸めの差）。**判定はどちらも `reject-new-run` で同じである。**

**読み。** **run1 は、どの見積もりでも、どの率でも通らない。**
いま買える最大は US$2.4936/h で **2.084 時間**、US$3.0567/h で **1.70 時間**である。
run1 の見積もりは 2.876 時間なので、**足りないのは 0.79 時間ぶんの上限である。**

**run1 完走 + 4-3 に要る額**（上限の判断材料。**決めるのは利用者である**）

| 前提 | run1 の h | run1 の率 | run1 後の予測 | ＋4-3（(b) 4.76 h × US$0.1067/h = US$0.508） | **必要上限**（＝予測 ÷ 0.90） |
| --- | ---: | ---: | ---: | ---: | ---: |
| 見積もりどおり・予備なし | 2.876 | 2.4936 | 114.473 | 114.980 | **US$127.76** |
| 同・M3 の SXM4 率 | 2.876 | 3.0566666666666666 | 116.092 | 116.600 | **US$129.56** |
| 予備 0.5 h を積む | 3.376 | 2.4936 | 115.719 | 116.227 | **US$129.14** |
| 予備 0.5 h・SXM4 率 | 3.376 | 3.0566666666666666 | 117.620 | 118.128 | **US$131.25** |

**いちばん緩い行で US$127.76、いちばん堅い行で US$131.25。**現行は US$125 である。

3 つ注意がある。

1. **4-3 の US$0.508 は下限である。**US$0.1067/h は Tesla P40 の `dph_total` で、disk 代を含まない。
   120 GB を要求して `storage_cost` が US$0.20/GB/月なら +US$0.033/h（4.76 h で US$0.664）、
   US$1.00/GB/月 なら +US$0.164/h（US$1.29）になる。必要上限はその差ぶん（最大 US$0.9 程度）上振れする
2. **run1 の 2.876 時間は、いまも実測ではない。**その内訳の 6 行は M3 の計画値のままで、
   [`m3r-stop-line.json`](../../experiments/tsukuyomi_ojousama/reports/m3r-stop-line.json) が
   「3.8 倍まで外れうる」と明記している。**run1 は 1 度この見積もりの内側で失敗している**
3. **予備を積まない見積もりで上限を決めると、同じ失敗を繰り返す。**予備 0.5 h の行と積まない行の差は
   US$1.4〜1.7 である

> **率はレンタル時に引き直す。** ここの数値は `dph_total` の仮定に依存する。
> `vastai show instance --raw` の `dph_total` と `storage_cost` を読み、
> `rate = dph_total + storage_cost × disk_gb / 730` を計算し、
> `--spent` にはその時点の `accrued_estimate.total` を入れて
> **借りる前にもう一度 preflight を通す**こと。

### 4-2: run1（再挑戦）— 学習と変換（2× A100 80GB **SXM4**、見積もり 2.876 時間）

**1 回目は失敗している。**この手順は、その失敗から 4 点を織り込んで書き直したものである。

1. **課金の前に、課金なしで済む段を全部通す**（段 0-1）—— run1 は課金を払いながら 1.25 h 止まったまま、
   手元では 0.01 秒で終わる処理を疑っていた
2. **SXM4 を名指しし、かつ実請求率を計算する**（段 0-2、段 a）—— この 2 つは独立である。破棄した 47239987 は **SXM4 だったが disk 代**で落ち、ハングした 44937484 は **価格は足りたが PCIe** だった。`tools/offer_check.py` が両方を一度に判定する
3. **本番の前に 2 step の smoke test を、時限つきで通す**（段 c）—— run1 は本番を直接起動した
4. **借りる前に実請求率を計算する**（段 0-2）—— 48911444 は検索値の 1.59 倍を請求してきた

#### 借りる前（課金ゼロ。**ここを飛ばさない**）

| # | 作業 | ゲート |
| --- | --- | --- |
| 0-1 | 出荷 parquet 70 行を `preprocess_function` に手元で通す | 70 例・shape (17,280) が出る。**実測 0.01 秒**（`--max_length` / `--min_length` の有無で差なし）。CPU だけで完結する段は、GPU の時間を買う前に必ず通す |
| 0-2 | offer を **SXM4 で名指しして**探し、**`tools/offer_check.py` にかける** | `vastai search offers 'gpu_name=A100_SXM4 num_gpus=2 gpu_ram>=79 …' --raw` の JSON を `check_offer(offer, spent=<accrued>, limit=112.5, planned_hours=<h>, num_gpus_needed=2)` に渡す。`usable=False` なら借りない。**2 つの条件は独立であることに注意** —— 破棄した 47239987 は SXM4 だが disk 代で落ち、ハングした 44937484 は価格は足りたが PCIe だった。SXM4 を名指しするだけでは前者を防げず、価格を見るだけでは後者を防げない |
| 0-3 | disk を仕事に合わせて要求する | 逐次変換すれば同時に存在する ZeRO state は最大 2 本で、**約 328 GB** で足りる（`vast-run` skill）。900 GB は storage_cost 次第で +US$1.2/h になる。**`--disk` は create 後に変更できない** |
| 0-4 | 打ち切り線を UTC 時刻で書き出す | [`STOP_LINE.md`](../../experiments/tsukuyomi_ojousama/m3r/STOP_LINE.md) の手順。**accrued は 107.301 から引く**（同文書の数値は 102.812 時点） |

#### 借りてから

| # | 作業 | 累積 | 締切オフセット | ゲート |
| --- | --- | ---: | --- | --- |
| a | rent + bootstrap（j-moshi-ext snapshot 約 15.76 GB の DL 込み） | 0.451 h | 0h27m | `bootstrap_m3_instance.sh` の host check が通る（2 枚・80,000 MiB・80 GiB RAM）。**gate は緩めない。**加えて **`nvidia-smi` の GPU 名に `SXM4` が入っていること。`PCIe` ならその場で破棄する**（48911444 と同じ判断。損失は US$0.2 程度） |
| b | base model dq16 を構築し、parquet を upload | 0.802 h | 0h48m | dq16 が **16,742,873,520 bytes**。train / dev の sha256 が `adea0474…` / `f21d821e…` |
| **c** | **2 step の smoke test（時限つき）と disk 検算** | 1.103 h | 1h06m | **下記** |
| d | **V-real 学習（45 step / 5 epoch）** | 2.177 h | 2h11m | 起動 assertion が **`Num examples 70`** / `Total train batch size 8` / `Total optimization steps 45`。**違えば即 kill** |
| e | ZeRO state 5 本を bf16 へ変換 | 2.677 h | 2h41m | 5 本すべての checksum 一致 |
| f | 84 GB を保管インスタンスへ push | 2.827 h | 2h50m | 転送先で checksum 一致 |
| g | **インスタンスを破棄** | 2.876 h | 2h53m | `vastai show instances` に該当 ID がない。台帳を精算する |

#### 段 c —— 本番の前に、この箱が学習できるかを数分で確かめる

**run1 が飛ばしたのはこの段である。**計画には「smoke test」と書いてあったが、
ゲートは「eval の内訳 4 項目が stdout に出る」だけで、**時限も、通らなかったときの手が無かった。**

| | |
| --- | --- |
| コマンド | 本番と同じ引数に **`--max_train_steps 2`** を足す（[`finetune.py:333`](../../finetune.py) で受け、1124 行で止まる）。環境変数に **`NCCL_DEBUG=INFO`** |
| ゲート 1 | 起動 assertion（`Num examples 70` / `Total optimization steps 45`）が **時限内に出る** |
| ゲート 2 | 2 step 走り、`LRs(next step)` が **0 → 目標の 1/2 → 目標**（下の「未決 2 → 決着」） |
| ゲート 3 | eval の内訳 4 項目が `--with_tracking` 無しで stdout に出る |
| ゲート 4 | ZeRO-3 checkpoint の実測サイズ × 本数 + 固定分 < 実効容量 |
| **時限** | **10 分。**`Num examples` が出なければ失敗として扱う |
| 通らなかったら | **`NCCL_P2P_DISABLE=1` を足して 1 回だけやり直す。**それでも 10 分で出なければ**この箱を諦めて破棄する。**単一 GPU への切り替えは試さない（133.94 GB 常駐に対し 1 枚 85.90 GB） |

**時限 10 分の根拠は実測ではない。賭けの大きさである。**
起動 assertion までの所要は M3 でも単独計測がない。一方 run1 の停止は **56 分と 19 分**続き、
どちらも 10 分をはるかに超えていた。10 分の待ちは US$2.4936/h で **US$0.42**、
run1 が失った US$4.289 の 1/10 である。**外れても安い側に倒す。**

**`NCCL_P2P_DISABLE=1` で通ったら、それは測定結果である。**
[`m3r-run1-failure.json`](../../experiments/tsukuyomi_ojousama/reports/m3r-run1-failure.json) の
`leading_hypothesis.status` は `未検証` であり、この 1 回で `検証済み` に変わる。**報告に書くこと。**

#### 変えていないこと

**run1 に生成は入っていない**（下の「未決 3」）。入れると 2.876 + 1.597 = 4.473 時間になり、
いま preflight が許す 2.084 時間（US$2.4936/h）を大きく超える。**上限を上げても、この配置は変えない。**

> **d のゲートは第3段 3-3 が直した。** 旧文言は `Num examples 72` で、
> 出荷 parquet（70 行）で走らせると**正しい run を kill する**ものだった。
> `ceil(70/8) = 9`、`9 × 5 = 45`。**総 step は 45 で M3 と揃う。行数だけが違う。**
> **run1 はこの assertion に一度も届いていない。**直したゲートは、まだ一度も使われていない。

### 4-3: run2 — export（安い箱、見積もり 約 3.07 時間 / US$0.16）

内訳は bootstrap 0.300 + 転送 2.718 + destroy 0.050 = **3.068 時間**、US$0.0525/h で US$0.161。
第3段の run2 見積もり 5.165 時間は **4-1 の forward 0.5 時間（完了済み）と生成 1.597 時間（配置未決）**を
含んでいたので、export だけならこの内訳になる。
**未決 3 は選択肢 (b) で決着している。**その場合この 4-3 は独立した run ではなく、
**20 GiB 級 1 本（生成 1.597 + 転送 2.718 + bootstrap 0.45 = 4.76 時間）に統合される。**
上の必要額の表はその前提（4.76 h × US$0.1067/h = US$0.508）で引いてある。

| # | 作業 | ゲート |
| --- | --- | --- |
| a | 安い箱を借り、run1 が push した 5 本を受ける | disk が 84 GB + 余裕を持つ。**`dph_total` は disk 代を含まない**（48911444 で確定）。`rate = dph_total + storage_cost × disk_gb / 730` を計算してから借りる（停止中でも disk は課金される） |
| b | 5 本 × 16.74 GB を手元へ転送 | 手元で checksum 一致。**1 本目が 40 分で終わらなければ残りを引き直す** |
| c | **インスタンスを破棄** | `vastai show instances` に該当 ID がない |

**export のサイズは仮定である。** 16.74 GB は base model dq16 の重みで、
**fine-tune 後の export サイズは M3 でも計測されていない**（M3 が checksum まで確認したのは
dq8 control の 15,375,500,136 bytes）。1 本目の実サイズで全体を引き直すこと。

### 未決 1: acoustic 損失重みを振るか — **決めない。状況は 2026-08-27 から動いていない**

**run1 が失敗したので、この判断に要る材料はひとつも増えていない。**
下の「決めるのに足りていないもの」は control の結果そのもので、
**run1 は学習に到達していないため、それは依然として存在しない。**
4-1 が数値を確定させた。**しかし M3-R は control であり、変数を足すかどうかは別の判断である。**
[この計画が答えられないこと](#この計画が答えられないこと)の「崩壊の原因が LR かデータ構造か」と
同じ理由で、ここで足すと**分離できない変数が 3 つ目になる。**

**判断に要る材料（すべて実測または実装の事実）**

| # | 材料 |
| --- | --- |
| 1 | **A-acoustic の寄与は audio_total の 1.3%**（4-1 実測）。名目の重み配分は 3.27%、[監査](./j-moshi-tsukuyomi-ojousama-m3r-dataset-audit.md) §3 の推定は 0.99%。3 つは分母が違うだけで、いずれも「声を運ぶトークンに 1〜3% しか載っていない」を意味する |
| 2 | **`--acoustic_loss_weight` は A 側と複製ヘッド側を分けられない。**[`finetune.py:689-712`](../../finetune.py) の `acoustic_scale` は両方に同じ値で掛かり、正規化の分母にも両方が入る。重みを上げると**未学習の複製ヘッドの acoustic（現状 3.0%）も同じだけ上がる**。分けるにはコード変更が要る |
| 3 | **audio 目的関数の 83.1% は複製ヘッドが占めている。** A 側の相対寄与を上げる手段は acoustic 重みだけではない。user stream 側の扱いを変える案は**一度も測っていない** |
| 4 | **予算。** 重みを振った腕を足すと run1 相当がもう 1 本要る。preflight の残余は **US$5.199** で、**いまは run1 を 1 本すら買えない。**2 本目の議論は、上限が動いて run1 が通った後の話である |
| 5 | **[監査](./j-moshi-tsukuyomi-ojousama-m3r-dataset-audit.md) §8 の手順 6 は「4-1 の結果を見て決める」と書いている。**その時点が今である |

**決めるのに足りていないもの**: control の結果そのもの。**まだ得られていない。**
条件4 の `closed` が何%になるかを見れば、「声に寄らなかった」のか「寄ったが足りない」のかが分かる。
**前者なら重みは第一の容疑者になり、後者なら M4 の変数として扱うほうが素直である。**
run1 が失敗した 2026-08-27 以降も、`closed` は**未測定**のままである
（[`m3r-run1-failure.json`](../../experiments/tsukuyomi_ojousama/reports/m3r-run1-failure.json)）。
**したがってこの未決は、run1 が通るまで前に進まない。**
上限の変更を伴う選択は利用者の決定であり、こちらが再承認するものではない。

### 未決 2 → **決着。`--num_warmup_steps 0` を明示する（実効 2 step）**

**これは未決ではなかった。分類が誤っていた。**
[実験計画](./j-moshi-tsukuyomi-ojousama-plan.md) の Stage 2 に、**候補が 1 件も存在しない時点で
事前登録された決定**がある。

> 過去 config から scheduler を回収できるまでは、control は現行実装の既定値である
> `--num_warmup_steps 0` を明示する。過去値が判明した場合は、その値を再現 run にのみ使用し、
> 別 run として記録する。

過去 config（`configs/j-moshi-ext-amitaro.yaml`）は回収不能である（`m0/artifact-recovery.md:40`）。
**したがって条件は満たされておらず、事前登録どおり 0 を明示する。**

**そして「0」は「warmup なし」ではない。** DeepSpeed の `WarmupLR` は
`warmup_num_steps=0` を `max(2, ·)` にクランプするので、**実効 2 step の linear warmup** になる。

〔実測で裏づけ済み〕クランプが真なら linear warmup の LR は
step1 = 0、step2 = 目標の 1/2、step3 = 目標、となる。M3 の実測ログ
（`train-v-real.nohup:773-775` の `LRs(next step)`）は
**0.000e+00 → 1.500e-05 → 3.000e-05** で、目標 3e-5 に対しこの予測と厳密に一致する。

| | 本数 | 45 step 目の LR |
| --- | ---: | --- |
| **M3-R（採用）= M3 と同じ** | **実効 2** | 目標の 100%（step 3 で到達） |
| リファレンス `examples/finetune_accelerate.sh` | 500 | 目標の **9%** |

**なぜリファレンスの 500 を採らないか。** 総 step 45 の 11 倍で、linear なら最終 step でも
目標 LR の 9% にしか届かない。**学習率を下げる決定（3e-5 → 2e-6 / 4e-6）と二重にかかり、
control が「LR を下げた run」なのか「warmup が長すぎた run」なのか区別できなくなる。**

**M3-R は control である。M3 と同じ warmup で走らせることが、比較可能性そのものである。**
45 step の run に適した本数は測られていないが、それは **M4 の変数**であって control の変数ではない。

**起動時の確認**: ログの `LRs(next step)` が step1 = 0、step2 = 目標の 1/2、step3 = 目標 になること。
ならなければ scheduler が想定と違う。

### 未決 3: 生成をどの箱で走らせるか — **決めない。材料を並べる**

**案 B の run1 に生成は入っていない。** 第3段の分割計画では生成は run2（単一 24 GB）にあったが、
その run2 は 4-1 の forward も含んでおり、**4-1 が済んだいま run2 の中身は export だけになっている。**
生成の置き場は決まっていない。

**判断に要る材料（実測）**

| # | 材料 |
| --- | --- |
| 1 | **生成には dq16 が載る箱が要る。** fp16 で **15.59 GiB**、CPU 側の fp32 ロードで **31.19 GiB**。4-1 の gate は **20 GiB カード / 40 GiB RAM** で、実際に借りたのは V100 32GB（US$0.3017/h） |
| 2 | **実在を確認した安い箱は 12 GB（RTX 3060）と 8 GB（GTX 1070）で、生成には載らない。** 転送の中継にはなる |
| 3 | **生成の見積もりは 1.597 時間**（control 50 prompt × 2 sample = 0.497 + 5 epoch × 50 prompt × 2 sample = 1.100）。いずれも M3 計画由来で、**単独の実測はない** |
| 4 | **run1 には入れられない。** 2.876 + 1.597 = 4.473 時間で、いま preflight が許す 2.084 時間（US$2.4936/h）を大きく超える |
| 5 | **run1 の後の残余は、上限の決定に依存する。**accrued 107.301 では run1 自体が通らないので、この計算は上限が動いてからでないと確定しない（上の必要額の表） |

**選択肢**

- **(a) 生成箱と転送箱を分ける（3 run）** — 生成は 20 GiB 級、転送は US$0.0525/h の箱。84 GB を 2 回動かす
- **(b) 20 GiB 級の安い箱 1 本で生成と転送を兼ねる** — 転送 2.718 + 生成 1.597 + bootstrap で 4.6 時間程度。
  US$0.3017/h なら US$1.39 になる。**率が US$0.19/h 以下の 20 GiB 級が要る**
  （この判定は 2026-08-27 時点の「run1 後の残余 US$0.897」を基準にしたものだった。
  run1 の失敗でその残余は消えたが、**安い箱を選ぶ理由は残っている**）
- **(c) control 生成を M3 の既存 550 件で代替する** — 0.497 時間の節約。ただし M3 の生成は
  **seed 未記録・prompt あたり 1 sample** で、この計画の「seed 固定・複数サンプル」と矛盾する

**未取得の材料**: 20 GiB 級カードの実勢価格、インスタンス間転送の実測速度、生成 1 本あたりの実測時間。
**いずれも `vastai search offers` と 1 本目の実測で埋まる。**

> **2026-08-27 追記 — 価格は埋まった。選択肢 (b) が成立する。**
> **（2026-08-28 追記: 箱の選び方としての結論は変わらない。ただし下の率は `dph_total` で
> disk 代を含まず、「run1 後の残余 US$0.897 に収まる」という予算の根拠は run1 の失敗で消えた。）**
> `vastai search offers` で 20 GiB 級・RAM 40 GB 以上・disk 120 GB 以上を絞ると 24 件あり、
> うち **US$0.19/h 以下が 2 件**（`m0/spend-ledger.json` の `offer_surveys` 第2回）。
>
> | GPU | VRAM | RAM | disk | US$/h | 下り |
> | --- | ---: | ---: | ---: | ---: | ---: |
> | **Tesla P40** | 24.0 GB | 63 GB | 726 GB | **0.1067** | 638 Mbps |
> | Q RTX 6000 | 22.5 GB | 47 GB | 299 GB | 0.1343 | 811 Mbps |
>
> 選択肢 (b) は 4.76 時間（生成 1.597 + 転送 2.718 + bootstrap 0.45）で、
> Tesla P40 なら **US$0.508**。選択肢 (a)（生成箱と転送箱を分ける）は US$0.581 で、
> **(b) より高い**。箱が 1 本増えるためである。**この比較は run1 の失敗に影響されない。**
>
> **US$0.508 は下限である。**US$0.1067/h は `dph_total` で disk 代を含まない。
> 120 GB を要求して `storage_cost` が US$0.20/GB/月なら実請求は約 US$0.140/h（4.76 h で US$0.664）、
> US$1.00/GB/月 なら約 US$0.271/h（US$1.29）になる。**借りる直前に計算し直すこと。**
>
> **したがって未決 3 の答えは (b) である。** 残る不確実性は所要時間の見積もり側にあり、
> 生成 1.597 h と転送 2.718 h はどちらも M3 計画由来で単独の実測がない。**run1 の実測で置き換えること。**
> offer は時々刻々変わるので、借りる直前に取り直すこと。

---

## 判定基準（第 4 段の完了条件）

第 1 段で改訂した基準を使う。**候補を見る前に固定してある**（2026-08-25、候補 0 件の時点）。
正本は [`m3/DATASET_SPEC.md`](../../experiments/tsukuyomi_ojousama/m3/DATASET_SPEC.md) である。

| # | 条件 | 基準 | 第3段までに分かっていること |
| --- | --- | --- | --- |
| 1 | dataset 一致 | 不一致 0 | **達成済み。** 9 種すべて 0、負の対照で 78/78 が落ちる |
| 2 | tokenize 台帳 | skip が全記録 | **達成済み。** skip 0 / rejected 0、tokenize の全フラグを sidecar と `m3r-tokenize.json` に記録。フラグを落とすと CI が赤くなることも確認済み |
| 3 | 崩壊なし | **音響指標を含む**新検出器で、退化が control 以下 | control 自身が general30 の 17/30 で退化している。**比較の相手は「崩壊していない基準線」ではない** |
| 4 | 話者らしさ | paired mean が正、かつ**較正帯までの距離を有意に閉じる**（区間推定つき、`closed >= 25%`）| 下の注記 |
| 5 | 明瞭度と turn-taking | 第 3 段で測った control 値を下回らない | control = **clean_transcribed_ratio 0.80**、median perplexity 2523.1 / 3285.2。**perplexity 単独では判定しない**（符号が逆になる）。`repetitive_transcripts` と `empty_transcript` を併記する |
| 6 | 採用理由 | 全 epoch の判断が記録されている | 全 epoch を export する設定にしてある |

**条件 3 と 4 はインターロックする**（`tools/likeness_guard.py`）。
崩壊した腕は、話者らしさがいくら高くても条件 4 を通せない。

**条件 3・5 の control 値についての注記。** 第3段が採点したのは **M3 が生成した control 腕**であり、
seed は記録されておらず prompt あたり 1 sample である。第4段が control を
**seed 固定・複数サンプル**で取り直すなら、**基準値もその新しい control で取り直す**
（同じ prompt set、同じ道具、同じ較正ファイル）。
取り直さずに新旧を混ぜて比べると、差が「学習の効果」なのか「抽選の回数」なのか分からなくなる。

### 条件 4 についての注記 — 基準は変えない

第3段の往復測定で、**Mimi を通す限り `closed = 1.0` は到達不能**であることが分かった。
対象話者の生録音 10 件をそのまま Mimi 8 codebook に通して戻すと ECAPA は
mean **0.8166 → 0.6607** に落ちる（−0.156）。生成 checkpoint は必ずこのコーデックを通るので、
**実質の天井は `closed = 64.9%`** である。

| | 帯 | control からの距離 | 25% が要求する値 |
| --- | ---: | ---: | ---: |
| 生録音（**この基準の分母**） | 0.8166 | 0.4438 | **0.4838** |
| Mimi 往復後（到達可能な天井） | 0.6607 | 0.2879 | 0.4448 |

**分母は生録音のままとする。25% も変えない。** 理由は 2 つある。

1. コーデック帯を分母にするとバーは 0.4838 → 0.4448 へ**下がる**。**ゲートが緩む方向の変更**である
2. その提案は decoded-A の 0.6505 という**候補の形をした数字を見た後に**出てきた。
   `CLAUDE.md` は「候補を見てから決めた閾値は、候補を通すために決めた閾値と区別がつかない」と定める

**到達可能性は確認済み。** 天井 0.6607 はバー 0.4838 を **+0.1769 上回る**。
コーデック帯の下限 0.5038 でさえ `closed = 29.5%` でバーを超える。
**基準を緩めなくても到達可能である。**
同じ内容が [`DATASET_SPEC.md` の条件4 節](../../experiments/tsukuyomi_ojousama/m3/DATASET_SPEC.md)（2026-08-27 追記）にあり、
この文書はそれと同じことを言っている。**食い違えば `DATASET_SPEC.md` が正である。**

---

## 走行中の運用（M3 の超過 3 原因への対策）

1. 打ち切り線を**経過時間ではなく UTC 時刻**で固定する。レンタル直後に一度だけ計算して
   ファイルに書き出し、以後は再計算しない。読むのは instance の `start_date` だけである
2. **export 時間を見積もりに含める**（実測 8.56 MB/s。単箱なら全体の 39%）。
   「学習が終わった」は run の中間点であって終点ではない
3. `tools/experiment_budget.py` を**走行中に 20 分ごとに呼ぶ**。
   非ゼロの終了コードは、ステップの途中でも止める合図である
4. **各ステップの開始と終了を UTC で記録する。** M3 のセッションは事後に分解できず、
   そのせいで見積もりの 9 行中 6 行がいまも計画値のままである
5. 4-1 はこの運用で走り、**1.50 時間の線に対し 0.382 時間（25%）で終わった**
6. **run1 もこの運用で走り、3.376 時間の線に対し 1.72 時間で自主中断した。線は守れている。**
   守っても成果がゼロだったのは、線の問題ではなく **(a) 借りた箱の種別と (b) smoke test を飛ばしたこと**による。
   **打ち切り線は「金を使いすぎない」ための計器であって、「無駄に使わない」ための計器ではない。**
   後者は借りる前の段（4-2 の段 0-1〜0-4）が受け持つ
7. **止めた理由を、止めたその場で書く。**run1 の中断判断（残り 1.78 h に対し仕事 1.57 h、猶予 0.21 h）は
   [`m3r-run1-failure.json`](../../experiments/tsukuyomi_ojousama/reports/m3r-run1-failure.json) に残っており、
   だからこの節を書き直せている

---

## この計画が答えられないこと

正直に書いておく。

| 項目 | なぜ答えられないか |
| --- | --- |
| 崩壊の原因が LR かデータ構造か | M3-R は**両方を同時に**直す。分離しない。control を得ることが目的だからである |
| acoustic 損失重みを上げたらどうなるか | **未決**（上の「未決 1」）。振れば 3 つ目の変数になり、振らなければ「声を運ぶトークンに 1.3%」のまま control を取ることになる。**どちらを選んでも、この run 単独では重みの効果は測れない** |
| 実音 vs 合成音の比較 | V-tts の教師が帯の外なので、この比較は成立しない。**M2 に戻らない限り答えられない** |
| live 対話での挙動 | マイクを持つ人間が必要。`CLAUDE.md` の「live 対話で崩壊する checkpoint は不合格」は未検証のまま残る |
| 更新範囲を狭めた場合 | 次の変数として保留。M3-R が通れば、そこから初めて比較できる |
| 学習投入音声が「つくよみちゃんに聞こえるか」 | 第3段は統計だけで通した。**耳では一度も聴いていない**（`m3r-roundtrip.json` の `limits`）。韻律・話速・アクセントはどの数値にも現れない |
| 相槌音声の話者一貫性 | 2-3 のゲートどおりの ECAPA 測定を行っていない。凍結 ref-wav と NCC が代わりの証拠である |
| **run1 がなぜ collective で止まったか** | **GPU を借りないと確かめられない。**仮説は NCCL P2P over PCIe で、**未検証**。手元にあるのは症状（CPU 100%・GPU-Util 100%・I/O ゼロ）と、M3 が SXM4 だったという差だけである。**4-2 の段 c が数分で答える** |

**M3-R が不合格でも、それは「原因が特定できた」ことにはならない。**
その場合の次の一手は更新範囲を狭めること（`--parameters_to_finetune tempformer`）であり、
過去のお嬢様 run が約 40 分・tempformer-only で成功している前例がある。

---

## 記録の突き合わせ（他ファイルに残る旧い値）

**この計画からは消したが、他のファイルにまだ残っている値**である。
いずれもこの文書の担当外なので直していない。**第4段に入る前に片付けること。**

| 場所 | 残っている値 | 実測 |
| --- | --- | --- |
| `reports/m3r-tokenize.json` の `parquet.frames` と `differences_from_m3` | train min 195 / max 299 / mean 240.833、重なり 1,204 | 出荷 parquet は **min 201 / max 302 / mean 244.571**、重なり **1,202**（`m3r-dataset-agreement.json`、`m3r-timeline.json`）。lead-in 0.3 の旧ビルドの値が残っている |
| `m3r/STOP_LINE.md` §6 B と `reports/m3r-stop-line.json` | 安い箱の率 **US$0.45/h**（仮定）、必要上限 US$126.70 | 実在するのは **US$0.0525/h**（RTX 3060 12GB。ただし `dph_total`）。必要上限は run1 の失敗後 **US$127.76〜131.25** に上がった（上の必要額の表） |
| ~~`m0/spend-ledger.json`~~ | ~~案 B の offer 探索結果が未記録~~ | **解消済み。** 台帳の `offer_surveys` に 2026-08-27 の 2 回分（RTX 3060 12GB US$0.0525/h、GTX 1070 US$0.0481/h、Tesla P40 24GB US$0.1067/h ほか）が記録されている |
| `m0/spend-ledger.json` の `accrued_estimate.method_appendix` | 追記が 48838452 の「102.697 → 102.812」で止まっている | `total` は **107.301** に更新済みで、`instance_48911444` 0.2 と `instance_48911872` 4.289 も入っている。**合計は正しく、追記だけが欠けている** |
| `m3r/STOP_LINE.md` §0 と `reports/m3r-stop-line.json` | accrued **102.812** を前提にした preflight 残余 US$9.803、run2 行「111.867 / 4.0 h @ US$0.45/h」 | accrued は **107.301**、preflight 残余は **US$5.199**。打ち切り線の**引き方**は有効だが、**数値は全部引き直しが要る** |
| `reports/m3r-stop-line.json` の `verdict` | 「見積もり 7.041 時間 US$21.52 に対し preflight が許す残余は US$9.803 = 3.207 時間」 | 残余は **US$5.199**。run1 の失敗（US$4.289）と 48911444（US$0.20）を含んでいない |
| ~~`.claude/skills/vast-run/SKILL.md`~~ | ~~accrued US$102.812 / 残余 US$9.688~~ | **解消済み。** accrued **107.301** / 残余 **US$5.199** に更新し、再挑戦が preflight で拒否されることも書いた。SXM4 と smoke test の教訓は commit `ae14898`、offer 判定ツールは `tools/offer_check.py` |
| [M3-R データセット監査](./j-moshi-tsukuyomi-ojousama-m3r-dataset-audit.md) §8 の手順 6 | 「4-1 の結果を見て決める」 | その時点は来たが、**決めるのに要る control が run1 の失敗で得られていない**（未決 1） |

---

## 進め方

第0〜3段は課金ゼロで完了した。4-1 も完了し、実績は **US$0.115** である。
**4-2 は 1 回走らせて失敗し、US$4.289 を失った。**

**いちばん上にある問いは予算である。**

> **現行 US$125 では run1 を再挑戦できない。**どの見積もり・どの率でも preflight は `reject-new-run` を返す。
> 再挑戦（run1 完走 + 4-3）に要る上限は **US$127.76 〜 US$131.25**（上の必要額の表）。
> **上限をどうするかは利用者の決定であり、この文書は必要額を書くだけで、決めない。**

上限が動かない場合に残る道は 2 つしかない。**案 S**（診断だけ買う。SXM4 で段 a〜c を走らせ、
「collective を抜けるか」だけを US$2.24〜3.74 で確かめる。checkpoint は得られない）と、
**案 D**（第4段をここで止める。control checkpoint は得られず M4 は Blocked のまま）である。

**上限が動いたときに、着手の前に決めるべきことは 1 つだけである**（残る 2 つは決着済み）。

1. **acoustic 損失重みを振るか**（未決 1）。振るなら run1 の起動コマンドが変わる。
   **ただし判断材料である control は、run1 が失敗したためまだ存在しない。**状況は 2026-08-27 と同じである
2. ~~**warmup を何 step 入れるか**~~ → **決着**。実験計画が事前登録した `--num_warmup_steps 0`（実効 2 step、M3 と同一）
3. ~~**生成をどの箱で走らせるか**~~ → **決着**（未決 3 の (b)）。20 GiB 級 1 本で生成と転送を兼ねる。
   **率は借りる直前に `dph_total + storage_cost × disk_gb / 730` で引き直す**

**そして、上限が動く前でも課金ゼロで進められることがある。**

- 4-2 の段 0-1（`preprocess_function` を手元で通す）は**済んでいる**（実測 0.01 秒）
- 段 c の smoke test の起動コマンドと時限を、**借りる前に書いておく**
- `STOP_LINE.md` と `m3r-stop-line.json` の数値を accrued 107.301 で引き直す（上の突き合わせ表）

**打ち切り線は守れている。**run1 は 3.376 h の線に対し 1.72 h で止まった。
**失ったのは線の外側ではなく、線の内側で「学習できない箱を、確かめずに使った」ことによる。**
次に金を使うときは、**いちばん安い検査から順に**並べること。
