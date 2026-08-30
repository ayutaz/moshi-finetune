# M3 検証記録 — 原因診断の再検討と、計器の監査

作成日: 2026-08-24

対象: [M3 実施報告](./j-moshi-tsukuyomi-ojousama-m3-report.md)

本文書は M3 報告書の**査読記録**であり、報告書を置き換えるものではない。
**報告書の判定（不合格）は維持される。** 訂正されるのは、報告書が書いた**原因の読み**と、
判定に使った**計器の妥当性**である。

確度は【確定】【推定】【未確定】で明示する。本セッションが自分で計測し直した値には〔実測〕を付す。
すべての数値は `file:line` または URL を伴う。

---

## 1. なぜこの調査をしたか

M3 は Voice **control** — 過去に動いた条件を再現し、M4 が比較できる壊れていない基準を作る工程だった。
結果は不合格である。報告書は原因を「データ量」と読み、4 つのやり直し案を提示した
（[m3-report.md](./j-moshi-tsukuyomi-ojousama-m3-report.md) §5, §6）。

この読みには次の GPU 支出が直結する。累計 US$102.697 に対し上限 US$100 で既に超過している
（`m0/spend-ledger.json` の `accrued_estimate.total` と `experiment_cap`）。
誤った原因診断に基づいて再承認を求めれば、超過の上に無意味な支出が乗る。

`CLAUDE.md` は 2 つの規律を定めている。「前提を測ってから積み上げよ」と
「ゲートが妥当な結果を却下し続けるならゲートを疑え」である。**どちらもこの局面のためにある。**

そこで、追加の GPU を一切使わずに答えられる問いだけを選んで検証した。ローカルに残っている
550 個の生成トークン `.npy`、学習に投入した parquet、両 run の nohup ログだけで、
以下の 4 点はすべて届く範囲にあった。

1. 学習ログの step 1 は何を測った値なのか
2. 学習データのトークン列に欠陥がないか
3. 判定に使った計器（崩壊検出器・話者類似度・turn-taking の基準線）は測りたいものを測っているか
4. 文献は 8.67 分という規模と `3e-5` という学習率について何を言っているか

---

## 2. 中心論点: base model の開始 loss は何を意味するか

### 2.1 step 1 は未学習 base の測定値である

【確定】DeepSpeed の `WarmupLR` は `warmup_num_steps=0` を `max(2, ·)` にクランプし、
`_take_model_step` は `optimizer.step()` の**後**に `lr_scheduler.step()` を呼ぶ。
`finetune.py:912-914` はその後に `param_groups` から LR を読むので、**印字される LR は
「次 step で使う LR」**である。観測列は 0 → 1.5e-5 → 3.0e-5 なので、適用 LR は
step1=0 / step2=0 / step3=1.5e-5。**step 1・2・3 の 6 データ点すべてが未学習 base の測定値**である。

| run | step | 適用 LR | total | text | audio |
| --- | ---: | ---: | ---: | ---: | ---: |
| V-real | 1 | 0 | 14.40770 | 7.58805 | 6.81965 |
| V-real | 2 | 0 | 14.77831 | 7.95037 | 6.82794 |
| V-real | 3 | 1.5e-5 | 14.90782 | 7.78343 | 7.12439 |
| V-tts | 1 | 0 | 14.25199 | 7.15953 | 7.09246 |
| V-tts | 2 | 0 | 15.47230 | 8.28279 | 7.18951 |
| V-tts | 3 | 1.5e-5 | 14.43132 | 7.29924 | 7.13207 |
| **pooled (n=6)** | | | **14.708** | **7.677** | **7.031** |

出典: `data/.../m3/instance-out/train-v-real.nohup:773-775`, `v-tts-all.nohup:779-781`

〔実測〕台本は両 arm で同一なのに step1 の text が 7.588 対 7.160 と食い違う。
これは印字値が **8 対話中 1 対話のサンプル**であることの直接証拠である
（`finetune.py:916-923` が gather されていない `total_loss.item()` を印字。
per-device batch 1・grad-accum 4・world_size 2）。

### 2.2 3 つの数値は互いに比較できない

【確定】`finetune.py:532-536` の `text_loss = mean(non_pad) + 0.5 × mean(pad)` は
**分母の異なる 2 つの平均の和**であり、単一の平均でも perplexity 化できる量でもない。

| 量 | 定義 | chance | base 実測 | chance 比 |
| --- | --- | ---: | ---: | ---: |
| `loss/text_total` | mean(non_pad) + 0.5·mean(pad) | 15.560 | 7.677 | 49.3% |
| `loss/audio_total` | 214 分の重み付き平均 | 7.625 | 7.031 | 92.2% |
| `loss/total` | 上 2 つの単純和 | 23.185 | 14.708 | 63.4% |

【確定】`--model_user_stream` 有効時、1 フレームあたり semantic 2 本・acoustic 14 本なので
`audio_weight = 214·T`。semantic の取り分は 200/214 = **93.46%**、
話者 B 側の取り分がちょうど **50.0%**、うち B の semantic 単独で **46.7%**
（`finetune.py:546-554, 561-569`）。

### 2.3 確定したこと / しなかったこと

**【確定】学習終了時の semantic perplexity は 1.8〜3.0 である。**
2048 通りのトークンを実質 2〜3 通りに絞り込んでいる。モデリングではなく暗記の水準であり、
held-out で無音になることと整合する。

**【未確定】base の audio 7.03 を「j-moshi-ext の予測能力」と読むことはできない。**
学習に使った `dep_q=16` モデルは公開 j-moshi-ext ではなく、`models/utils.py:8-56` の
`extend_moshi_modules_for_user_stream` が depformer head 0-7 を `deepcopy` して 8-15 を作った
**手術済みモデル**である（`m3-instance-bootstrap.json` の `base_models.train-dq16.built_with`）。
audio 重みの 46.7% がこの未学習複製ヘッドに乗る。cross-entropy は上に非有界なので、
B 側が自信を持って外していれば A 側の実力は 6.0 nats かもしれないし 1.6 nats かもしれない。
**総和からは分離できない。**

**【確定】その内訳は計算された上で捨てられた。** `finetune.py:538-539, 571-579` は
`loss/text_non_pad`, `loss/audio_semantic`, `loss/audio_semantic_user` を毎 step 計算しているが、
`finetune.py:924` の `if args.with_tracking:` の内側でしか出力されない。
M3 は W&B を使っていないため、2 arm × 5 epoch = 10 回すべて破棄された。
〔実測〕両 nohup で `Evaluation at step` は **0 行**。**M3 には held-out loss が 1 つも存在しない。**

**【推定】「6.88 → 1.02 は崩壊の署名」という読みは成立しない。**
過去の失敗 depformer-only の始点 6.88 / 終点 1.02（`plan.md:93`）は M3 の 6.82 / 1.03 とほぼ一致する。
しかし `ln(2048)=7.625` から始まる以上、**成功 run も失敗 run も未学習時はほぼ chance から始まる**
（過去の成功 run も 12.53 → 1.24、`plan.md:84`）。
**始点も終点も、成功 run・失敗 run・M3 の 3 者を区別しない。**

---

## 3. データパイプラインの監査結果

### 3.1 text stream に既知要件の脱落がある【確定】

`tools/tokenize_text.py:59-64` は `--no_whitespace_before_word` が未指定のとき、
pyopenjtalk が切った全語の先頭に半角空白を足す。日本語では SentencePiece の結合が壊れる。

**このリポジトリ自身が 2 箇所でこれを禁じている。**

- `README-ja.md:109`「日本語や中国語など，単語間にスペースがない言語の場合
  `--no_whitespace_before_word` フラグを使用してください」
- `README.md:112` 同旨

M3 の `m3/DATASET_SPEC.md` はこのフラグに一言も触れていない。

〔実測〕出荷済み parquet の話者 A text row（`arr[0]`）を直接集計した。

| arm | frames | pad (id 3) | 非 pad | 裸 `▁` (id 9) | EPAD (id 0) | 実語 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| v-real | 18,841 | 14,433 (76.6%) | 4,408 | 1,416 (**32.1%**) | 1,236 (28.0%) | 1,756 (**39.8%**) |

〔実測〕さらに、実際の pyopenjtalk 分割で両経路を再現した。

| 経路 | token 総数 | 裸 `▁` |
| --- | ---: | ---: |
| 空白あり（現行の既定） | 3,226 | 1,469 (**45.5%**) |
| 空白なし（README-ja が指示） | 1,408 | 51 (3.6%) |

**text stream は必要の 2.3 倍のトークンを運び、その 45.5% が純粋な空白マーカーである。**
id 9 が `▁` であることは j-moshi-ext 同梱の `tokenizer_spm_32k_3.model` の
`id_to_piece(9)` で確認した。

base text loss 7.677 は total の 52.2% を占めるので、影響は無視できない。
ただしこれは **17 行のうち row 0 だけ**の欠陥であり、audio 側の崩壊軌跡を説明しない。

【未確定】過去に成功したあみたろ run が同じフラグを落としていたか。
`configs/j-moshi-ext-amitaro.yaml` は回収不能（`m0/artifact-recovery.md:40`）。
落としていたなら成功例と失敗例に共通する定数であり、M3 固有の失敗を説明しない。

### 3.2 タイムラインが「無音」を教えている【確定】

`tools/assemble_dialogue.py` は固定 lead-in 0.5 秒 / ターン間 0.4 秒 / **重なりなし**を実装する。
台本は全 80 対話が B-A-B で、話者 A は対話の中央で 1 回しか話さない
（`m3/scripts/dialogues-v1.jsonl`）。

結果として、**text が pad であることと話者 A が無音であることが 98% の精度で同値**になる。
モデルには「pad ⇒ 無音を出す」という決定的な近道がある。

V-real epoch4 以降で held-out の有声フレーム中央値が 0 になった現象は、
「loss が下がったのに壊れた」ではなく「**loss を下げる最短経路が無音だった**」で説明できる。

### 3.3 崩壊検出器が音を見ていない【確定】

`tools/dialogue_collapse.py` の docstring 自身が「Row 0 is the text stream」と書き、
`silent` / `monologue_loop` / `exact_repeat_collapse` はすべて row 0 から算出される。
**音響側の崩壊は原理的に検出できない。**

〔実測〕550 個の生成トークン `.npy`（shape `(17,124)`）から、話者 A の audio codebook 0
（row 1）を集計した。`deg` は distinct ≤ 8 かつ最頻占有率 ≥ 0.7 の clip 数
（本調査が探索的に置いた規則。M3-R 第 1 段が較正から導いた規則は entropy ≤ 1.43 bit かつ
distinct ≤ 9 で、control の general30 を **17/30** と判定する。以後は較正済みの 17 を正とし、
本表の 16 は探索時の値として残す。閾値は `reports/m3-collapse-acoustic-calibration.json`）。

| arm | prompt set | n | distinct 中央値 | deg |
| --- | --- | ---: | ---: | ---: |
| **control** | **general30** | 30 | **4** | **16** |
| control | held-out | 10 | 57 | 0 |
| control | seen | 10 | 65 | 0 |
| v-real/epoch2 | held-out | 10 | **81** | 0 |
| v-real/epoch4 | held-out | 10 | **5** | **5** |
| v-real/epoch5 | held-out | 10 | 5 | **8** |
| v-tts/epoch2 | held-out | 10 | **68** | 0 |
| v-real/epoch2 | general30 | 30 | 55 | **0** |
| v-tts/epoch2 | general30 | 30 | 49 | **0** |

この表から 3 つの事実が出る。

**【確定】基準線 control は general30 で壊れている。**
30 会話中 **16 会話**で distinct がちょうど 4、最頻トークンが 124 フレーム中 113 回。
〔実測〕そのトークン `1316` が無音であることを独立に確認した — 話者 B が教師強制で無音の
held-out prompt set において、user codebook 0 の **91.1%** が `1316` である。

同じ control は held-out / seen では distinct 57 / 65・退化 0/10 で健全である。
にもかかわらず `m3-collapse.json` の control `silent_count` は **0** と記録されている
（退化した 17 件の `emitted_text_ratio` は 0 にならないため）。

**【確定】劣化は epoch に対して単調である。** held-out の distinct 中央値は
v-real で 48 → 81 → 52 → **5** → 5 と推移する。
固定された形式欠陥は「水準」を決められても「傾き」を決められないので、
これは text tokenize の欠陥だけでは説明できない。
**V-real epoch4 の崩壊は指標のアーティファクトではなく実在する。**

**【確定】両 arm とも epoch 2 が頂点である。** v-real/epoch2 は held-out の distinct 81 で
control（57）を上回り、general30 で退化 0/30（control は 17/30）。
そして **epoch 2 は export されていない**。

### 3.4 検証して問題がなかった箇所【確定】

以下は仮説として検討され、証拠により否定された。M4 の調査対象から外してよい。

| 検査対象 | 結果 | 出典 |
| --- | --- | --- |
| stream 17 行の内訳（text 1 + Moshi 8 + user 8） | 正しい | `utils/data.py:21-27` |
| delay pattern の学習/推論一致 | 同一関数・同一パラメータ | `utils/data.py:32-64`, `generate.py:207` |
| チャンネル割当（A = つくよみ = Moshi） | 正しい | `tools/tokenize_audio.py:69-70` |
| Mimi 設定（24kHz / 12.5Hz / 8 codebook） | 学習と decode で同一 | `tools/tokenize_audio.py:53-56` |
| text tokenizer | j-moshi-ext 同梱と一致、pad=3 / EPAD=0 も整合 | `tokenizer_spm_32k_3.model` |

**audio 側のパイプラインに形式破綻はない。**

---

## 4. 文献が言っていること

### 4.1 話者適応の必要データ量 — 8.67 分は不足ではない

| 系 | データ量 | 結果 | 出典 |
| --- | --- | --- | --- |
| YourTTS | 20〜61 秒 | 20 秒で Sim-MOS 2.77 → 4.43 | [arXiv 2112.02418](https://arxiv.org/abs/2112.02418) |
| XTTS | 約 10 分 | SECS 0.5852 → 0.7166 | [arXiv 2406.04904](https://arxiv.org/html/2406.04904v1) |
| AdaSpeech2 | 5 分（decoder のみ） | 同等品質 | [arXiv 2104.09715](https://arxiv.org/abs/2104.09715) |
| VoiceTailor | 5〜10 秒（全体の 0.25%） | 同等の話者適応 | [arXiv 2408.14739](https://arxiv.org/abs/2408.14739) |

**【確定】「8.67 分では原理的に不可能」は文献では支持されない。**

ただし決定的な但し書きがある。上記はすべて**話者条件付け機構を持つモデルへの部分適応**である。
Moshi 系は公式 model card が "trained to produce only one voice to avoid impersonation" と述べる
**固定単声アーキテクチャ**で、声を変える経路が全パラメータ更新しかない。

**言えるのは「8.67 分は原理的に不可能」ではなく「8.67 分に対して `3e-5` の全パラメータ更新は
不可能」である。**

### 4.2 学習率 — `3e-5` は fine-tuning の値ではない

| 系 | 段階 | Temporal LR | Depth LR | 出典 |
| --- | --- | ---: | ---: | --- |
| Moshi | pre-training | 3e-5 | 2e-4 | [Moshi 論文](https://kyutai.org/Moshi.pdf) Table 1 |
| Moshi | instruct fine | 2e-6 | 2e-6 | 同上 §4.3 |
| J-Moshi | **事前学習** | **3e-5** | — | [NLP2025 D8-6](https://www.anlp.jp/proceedings/annual_meeting/2025/pdf_dir/D8-6.pdf) |
| J-Moshi | **fine-tuning** | **2e-6** | **4e-6** | 同上 |
| Kyutai 公式推奨 | fine-tuning | 2e-6（LoRA 既定） | — | [moshi-finetune](https://github.com/kyutai-labs/moshi-finetune) |
| **M3** | fine-tuning | **3e-5** | **3e-5** | `plan.md:87` |

【確定】M3 が使った `3e-5` は **J-Moshi の事前学習の学習率**であり、かつ
`finetune.py:166-176` の argparse 既定値そのものである。warmup も既定 0。

`plan.md:87` は「論文由来の小さい fine-tuning 学習率だけを基準にせず」と明示的に文献値を
脇に置いているが、その「過去に動いた control」の値は、文献に対しては事前学習の設定である。
**報告書の案 B（`1e-5`）でも文献値の 2.5〜5 倍である。**

【確定】Kyutai 公式の既定手法は **LoRA**（rank 128）であり、全パラメータ更新は opt-in。
ただし本リポジトリに LoRA 実装は存在しない（`grep -ri lora` で 0 件）。

### 4.3 崩壊の機序 — 既知現象の重なり

| 機序 | M3 との対応 | 出典 |
| --- | --- | --- |
| silence collapse — 無音が多数派のデータでは方策が always-SILENT へ崩壊 | チャンネル A の 69% が無音、B-A-B 固定 | [arXiv 2605.05626](https://arxiv.org/html/2605.05626) §3.4 |
| 破滅的忘却のべき乗則 — 更新パラメータ数と step 数の同時べき乗則。**early stopping では回避不能** | 45 step / 8.37B 全パラメータ | [arXiv 2401.05605](https://arxiv.org/html/2401.05605v1) §4.1 |
| neural degeneration — 反復の確率が反復ごとに上がる正のフィードバック。低 perplexity ≠ 高品質 | text pad 0.044 → 0.919 | [arXiv 1904.09751](https://arxiv.org/pdf/1904.09751) §4.2 |
| oversmoothing — 自己回帰モデルが空・短い系列に不当な確率を割り当てる | 有声フレーム中央値 0 | [arXiv 2112.08914](https://arxiv.org/abs/2112.08914v1) |

【確定】用語の訂正が要る。「mode collapse」は GAN 固有、「posterior collapse」は VAE 固有の現象で、
teacher forcing された自己回帰 LM の無音化とは機序が違う。
**M4 以降は degenerate / silence collapse または oversmoothing と呼ぶ。**

~~【確定】Kyutai 公式 README は「`duration_sec` を下げるとモデルがより早く無音になりうる」と警告し、
推奨系列長を 100〜300 秒としている。~~ 〔実測〕M3 の平均は **21.0 秒**
（v-real train 18,841 frames / 72 対話 / 12.5Hz）。
~~**最重要症状が、公式が名指しで警告する副作用と一致する。**~~

> **撤回（2026-08-25）**: 「推奨 100〜300 秒」は README の誤読である。全文検索して `300` は
> 一度も現れず、`duration_sec` は上限の指定であって長さを作る指定ではない。このプロジェクトで
> 唯一動いた過去の成功 run は **19.02 秒**（100 対話 / 31.7 分 / 1 対話 = 1 学習例）であり、
> M3 の 21.0 秒はそこから外れていない。**系列長は M3 の失敗の証拠にならない。**
> [M3-R データセット監査 §2](./j-moshi-tsukuyomi-ojousama-m3r-dataset-audit.md)。

### 4.4 対策には文献上の順位がある

| 対策 | 効果 | 出典 |
| --- | --- | --- |
| experience replay | 単独最強（14.3% → 66.3%） | [arXiv 2505.17496](https://arxiv.org/html/2505.17496) Table 1 |
| replay + merging 併用 | 68.0〜68.7% | 同上 |
| LoRA / 部分凍結 | 忘却が少ない。full FT は 10-100 倍 rank の摂動を学ぶ | [arXiv 2405.09673](https://arxiv.org/pdf/2405.09673) |
| model merging 単独 | 弱い（4.0〜19.3%） | [arXiv 2505.17496](https://arxiv.org/html/2505.17496) |
| **epoch 削減単独** | **べき乗則により回避不能** | [arXiv 2401.05605](https://arxiv.org/html/2401.05605v1) |

**報告書の案 A（epoch 削減）は文献上もっとも弱い手である。**

### 4.5 評価指標の妥当性

| 指標 | 妥当性 | 出典 |
| --- | --- | --- |
| ECAPA cosine（system レベル） | 主観話者類似度と r = 0.82〜0.86 | [VCC2020](https://www.isca-archive.org/vccbc_2020/das20_vccbc.pdf) Table 4 |
| ECAPA cosine（utterance レベル、合成音声） | LCC 0.512（R² ≈ 0.26） | [Interspeech 2024](https://www.isca-archive.org/interspeech_2024/ahn24b_interspeech.pdf) |
| 発話長交絡 | 同一話者 short-vs-long で EER 0.50 → **0.34** | [arXiv 2507.02176](https://arxiv.org/pdf/2507.02176) Table 2 |
| 動的特徴 | 話速・韻律・有声区間長を符号化しない | 同上 Fig.1 |
| ground-truth 較正帯 | VALL-E は明示的に upper bound と呼ぶ。prompt 3s→10s だけで 0.546 → 0.620 | [arXiv 2301.02111](https://arxiv.org/pdf/2301.02111) Table 6 |
| 参照不要の明瞭度 | ASR 出力に対する外部 LM の perplexity。**J-Moshi 自身が採用** | [Interspeech 2025](https://www.isca-archive.org/interspeech_2025/ohashi25_interspeech.pdf) §4.2 |

**【確定】条件 4 の効果量閾値 +0.02 は、測定系の交絡より小さい。**
VALL-E の Ground-truth 行は prompt 長 3s→10s だけで +0.074 動く。
M3 の有声フレーム中央値は control 125 に対し v-real/epoch1 が 9、epoch2 が 228 と 25 倍の開きがある。
**この閾値では真の声質転移と発話長の偶然を区別できない。**

**【確定】明瞭度は測定可能である。** J-Moshi 論文自身が、生成音声を Whisper にかけ、
その転写に対する日本語 LM の perplexity を fluency 指標として報告している。
参照テキストも参照音声も不要。報告書の「測定手段なし」は撤回できる。

---

## 5. 話者類似度 — 報告書がもっとも強く誤っている箇所

### 5.1 効果量基準は満たされていた【確定】

`m3-speaker-likeness.json` の `results_held_out` を集計した。事前登録の効果量基準は **+0.02**。

| arm | 勝ち | 採点可能 | full_set | survivors |
| --- | ---: | ---: | ---: | ---: |
| v-real/epoch5 | 1 | **1** | +0.0246 | +0.2463 |
| **v-tts/epoch3** | 5 | **9** | **+0.0321** | +0.0357 |
| v-tts/epoch4 | 4 | 5 | +0.0666 | +0.1332 |
| v-tts/epoch5 | 5 | 6 | +0.0628 | +0.1046 |

**4 腕が数値上は効果量基準を満たしている。** うち v-real/epoch5 は採点可能 1 件の見せかけで、
報告書自身がそう指摘している。しかし **v-tts/epoch3 は 10 件中 9 件が採点可能**で、
効果量 +0.0321 は基準を超えている。

**落ちたのは一貫性基準 8/10 だけである。**

### 5.2 8/10 は検出力を持たない【確定】

片側符号検定で α=0.0547、真の勝率 0.70 でも検出力 **0.383**、0.80 でも 0.678。
10 腕を同じ 10 文で判定しているので family-wise の偽陽性上限は 0.43。

**「効果なし」と「検出力不足」が分離できていない。**

### 5.3 較正帯を初めて測った〔実測〕

`tools/speaker_similarity.py` と同一の経路（ECAPA / voiced 抽出 / RMS 正規化）で、
対象話者の test 分割 10 録音の **leave-one-out** 類似度を測った。

| | mean | median | min | max |
| --- | ---: | ---: | ---: | ---: |
| **実話者の較正帯** | **0.8166** | 0.8175 | **0.7405** | 0.8780 |

`CLAUDE.md` は「0.74 の within-group 類似度は、実人間が 0.70 を取ると知るまで何も意味しない」と
書いている。**M3 の条件 4 はこの較正帯を持たずに判定されていた。**

### 5.4 V-tts の教師音声そのものが帯の外にある〔実測〕【確定】

同じ centroid に対し、V-tts の話者 A 教師音声 80 ターンを測った。

| | mean | median | min | max | 帯内 (≥0.7405) |
| --- | ---: | ---: | ---: | ---: | ---: |
| **V-tts 教師音声** | **0.7213** | 0.7267 | 0.6130 | 0.8167 | **35 / 80** |

**80 ターン中 45 ターンが対象話者の帯の外にある。**

報告書 §5 の「モデルが耐えたのは合成であることではなく、その均質さかもしれない」という留保は
足りない。**第三の、より重い差がある — V-tts 腕は学習前から天井が対象話者に届いていない。**
この腕で対象話者に到達することは原理的に不可能だった。

---

## 6. 報告書の原因診断は比較対象の取り違えである【確定】

報告書 §5 は「あみたろ実験 189.9 分の 4.6%」を主因とする。これは誤りである。

`plan.md` の同じ表の中で、2 つの数値は別の行に置かれている。

| `plan.md` の行 | 項目 | 値 | 何のデータか |
| --- | --- | ---: | --- |
| `:75` | 話者原音 | **189.9 分** | **Irodori-TTS の学習コーパス** |
| `:77` | 合成対話 | **31.7 分** | **過去の Moshi 学習データ** |

M3 の Moshi 学習データは `m3-dataset-agreement.json` で v-real **27.83 分**。
**like-for-like の比較は 27.83 / 31.7 = 87.8% であって、4.6% ではない。**

さらに、過去の成功 run は `plan.md:83-84` で **65 steps・loss 1.24** まで下げて崩壊していない。
M3 は 45 steps・1.72886 である。**過去の方が多く最適化して低い loss に着地し、壊れていない。**

「この量に対して 5 epoch・`3e-5` は強すぎる」は、**このプロジェクト自身の記録と矛盾する。**

---

## 7. 3 つの立場とその裁定

| | A: 前提が壊れている | B: 過学習と崩壊、それだけ | C: そもそも問える実験でなかった |
| --- | --- | --- | --- |
| 核心 | 計測系・基準線・text 符号化が壊れており、結論も 4 案も採用できない | 劣化は epoch に単調で両 arm に独立再現。次は更新範囲を狭めること | 主要判定条件が帰無時に解釈不能と計画自身が事前登録し、2 腕は音源以外に 4 つの未統制差を持つ |
| 最強の証拠 | 発行 text token の 45.5% が裸の `▁` | held-out の distinct が 81 → 5 と単調 | V-tts 教師が実話者帯の外（35/80 のみ帯内） |
| 自認する弱点 | audio 重みの 46.7% が手術由来。base loss を形式破綻の証拠にできない | 「1.02 と 1.03 の一致」は chance から始まる全 run に共通で情報量が薄い | 「実行すべきでなかった」は事後の視点 |

**衝突点 1: base audio 6.82 は何の証拠か。** → **B が正しい。** dq16 手術の既定路線であり、
形式破綻の証拠にはならない。立場 A 自身がこれを認めている。

**衝突点 2: text 欠陥はどこまで届くか。** → **B が正しいが、A も正しい。**
audio codebook の distinct が 81 → 5 へ落ちる軌跡は text 欠陥では説明できない。
ただし text 欠陥は本物であり、base loss の 52.2% を占める側の意味を変える。**対象が違う。**

**衝突点 3: V-tts が耐えたのはなぜか。** → **C が正しい。** 教師そのものが帯の外である。

**衝突点 4: 次の一手。** → **3 つとも部分的に正しく、順序が違うだけである。**

---

## 8. 総合判定

**M3 の「不合格」という判定は維持してよい。**
**しかし報告書の「原因の読み」と「声は寄らなかった」と turn-taking の解釈は採用してはならない。**

### 確定したこと

1. **崩壊は実在し、epoch に依存する。** held-out の distinct 中央値は v-real で 48 → 81 → 52 → 5 → 5。〔実測〕
2. **原因診断は比較対象の取り違え。** like-for-like は 87.8% であって 4.6% ではない。
3. **過去の成功 run は M3 より強く最適化して崩壊していない。** 65 steps / 1.24 対 45 steps / 1.73。
4. **turn-taking の基準線は壊れている。** control は general30 で 17/30 が退化状態。〔実測〕
5. **崩壊検出器は row 0 しか読まない。** `m3-collapse.json` の control `silent_count = 0` は上記を見落とす。
6. **条件 4 の効果量基準は満たされていた。** 落ちたのは一貫性基準 8/10 のみ。
7. **8/10 は検出力 0.383。** 「効果なし」と「検出力不足」が分離できていない。
8. **text stream の非 pad の 6 割がノイズトークン。** 既知要件 `--no_whitespace_before_word` の脱落。〔実測〕
9. **V-tts の教師音声が対象話者の帯の外（35/80）。** この腕の天井は学習前から届いていない。〔実測〕
10. **`3e-5` は J-Moshi の事前学習の値。** fine-tuning の文献値は 2e-6 / 4e-6。
11. **計測系:** 印字 loss は 8 対話中 1 対話。eval loss は 1 つも存在しない。seed 未記録・単発抽選。
12. **予算超過。** US$102.697 対 上限 US$100。

### 未確定のまま残ること

1. base audio 7.03 の話者 A 側 / B 側の内訳（重みの 46.7% が手術由来。内訳は破棄済み）
2. text 欠陥が失敗にどれだけ寄与したか（修正版 parquet との diff が要る）
3. 過去のあみたろ run が同じフラグを落としていたか（設定ファイル回収不能）
4. V-real epoch4/5 の無音が Moshi Appendix D の silence か background noise か
5. 崩壊の原因が LR か、データ構造か、更新パラメータ範囲か（M3 は 3 つを同時に振っている）
6. 観測された正の delta が本物の声質転移か、ECAPA が平坦な出力を減点しないことによる見かけか
7. control 17/30 の退化が general30 の prompt 形状の産物か J-Moshi-ext の性質か

### 記録間の不一致（1 件・~~未解消~~ → 解消済み）

`m3-speaker-likeness.json` は v-tts/epoch3 を `scorable: 9, higher_on: 5` と記録する。
本調査の一部の再測定は同じ腕を「10 件すべて採点可能、paired mean +0.0738」としており、
採点可能数と delta の両方で食い違う。
**出荷済み JSON の値を正とし、再測定値は参考として扱う。**
~~差の原因（有声区間の抽出条件か floor 判定か）は未特定であり、条件 4 を再設計する際に必ず解消すること。~~

> **解消（2026-08-25）**: 原因は**そのどちらでもなかった**。`likeness_guard.apply_degeneracy_guard` が
> clip 4 を `exact_repeat_collapse` として候補側から除いており、再測定はその guard を通らない経路だった。
> 同じ出荷ファイルの census が 10/10 採点可能・`below_floor: 0` と記録しているので、floor 由来ではありえない。
> 記録は [`m3-likeness-calibration.json`](../../experiments/tsukuyomi_ojousama/reports/m3-likeness-calibration.json)
> の `record_reconciliation`、経緯は [M3-R 実行計画](./j-moshi-tsukuyomi-ojousama-m3r-plan.md) 1-5。

---

## 9. M3 報告書に対する訂正

| # | 箇所 | 訂正 |
| --- | --- | --- |
| 1 | §5 原因の読み（および `plan.md:181`, `m3-plan.md:48,229,257`, `milestones.md:446`） | 189.9 分は TTS 学習コーパス、Moshi 学習データは 31.7 分。like-for-like は 87.8%。**段落全体を撤回し「原因未特定」と書き直す** |
| 2 | §3「5/10 にとどまる。ほぼコイン投げ」 | 効果量基準は満たされていた。**絶対値・較正帯・per-clip 分散を併記し「一貫性基準が構造的に達成困難だった」と書き直す** |
| 3 | §3「fine-tuning はモデルを『より答えるように』した」 | control は general30 で 17/30 が退化。**「比較の基準線が成立していない」と書き直す** |
| 4 | 条件 3 の判定 | `dialogue_collapse.py` は row 0 のみを読む。**全数値は「テキスト崩壊」としてしか読めないと明記** |
| 5 | 「6.88 → 1.02 の再現」 | 始点も終点も成功 run・失敗 run を区別しない。**loss から崩壊を推定する記述を撤回し、証拠を生成トークン側（distinct 81 → 5）に置き換える** |
| 6 | §3「11.39 → 1.73 へ単調に減少」 | epoch 平均でのみ真。step レベルでは約 4 割の遷移で増加。各値は 8 対話中 1 対話。**「epoch 平均は」と限定する** |
| 7 | §5「均質さかもしれない」 | V-tts 教師は帯の外（35/80）。**「この腕で対象話者に到達することは原理的に不可能だった」と書き直す** |
| 8 | §7「明瞭度 / 測定手段なし」 | J-Moshi 論文自身が参照不要の方法を使う。**「未実施。方法は存在し GPU なしで実行可能」と書き直す** |
| 9 | §6 やり直し案 | 案 C の根拠は撤回。案 A は文献上もっとも弱い。案 B の 1e-5 でも文献値の 2.5〜5 倍。**表を全面改訂する** |
| 10 | §7「満たせなかったもの」 | **「再現性の欠如」を 4 項目目として追加**（seed 未記録・prompt あたり 1 サンプル） |

---

## 10. 次にすべきこと

GPU を使わないものを先に置く。**1〜3 は blocking** で、これを済ませずに次の GPU 支出を求めてはならない。

| # | 作業 | コスト | 何が決まるか |
| --- | --- | --- | --- |
| **1** | 報告書を §9 の 10 件で訂正し、他 3 文書に伝播した「4.6%」「1/24」も直す | US$0 / 40 分 | 誤診に基づく次の支出判断が止まる |
| **2** | `--no_whitespace_before_word` 付きで parquet を作り直し、行単位で diff。row 1-16 が不変であることを確認 | US$0 / CPU 15 分 | text 欠陥の到達範囲が確定する |
| **3** | 条件 4 を再設計。較正帯・絶対 cosine・per-clip σ を必ず出力し、判定を「8/10 の符号検定」から「paired mean + 区間推定」へ。§8 の記録間不一致も解消 | US$0 / CPU 1 時間 | 「効果なし」と「検出力不足」が分離される |
| 4 | `dialogue_collapse.py` に音響指標を追加し、閾値を calibration に固定して全 11 腕 × 3 set を再判定 | US$0 / 45 分 | control が基準線として使えるかが自動判定される |
| 5 | tokenize の全フラグを manifest に記録し、parquet の text 統計を `tests/test_experiment_assets.py` のゲートにする | US$0 / 30 分 | 既知要件が再び黙って落ちるのを防ぐ |
| 6 | `finetune.py:916-923` を修正: gathered 平均を印字、loss 内訳 4 項目を `--with_tracking` なしで stdout へ、eval loss を smoke test で確認 | US$0 / 30 分 | M3 で 10 回破棄された内訳が二度と失われない |
| 7 | M4 のデータ設計を作り直す: ターン数を過去 run 相当へ、重なりを入れる、非発話チャンネルにルームトーンを入れて pad ⟺ 無音 の同値を切る、~~系列長を 100 秒以上に連結~~（**撤回。**§4.3 の誤読に立っていた） | US$0 / 数時間〜1 日 | 「loss を下げる最短経路が無音」という近道が消える |
| 8 | 明瞭度を測る。held-out と seen の 110 件を Whisper にかけ、転写の LM perplexity を取る。反復検出と併記 | US$0 / CPU 数時間 | 条件 5 が初めて数値化される |
| 9 | 学習投入 token を Mimi で decode して WAV に戻し、統計と耳の両方で確認 | US$0 / MPS 20 分 | パイプラインが端から端まで検証される |
| 10 | **（提案のみ）** forward 1 回で base loss の内訳を測る。bf16 16.74GB で単一 24GB カードに収まる | **US$0.20〜0.40** / 新上限の承認が前提 | base audio 7.03 の帰属が確定する |
| 11 | **（提案のみ）** 再走。LR を 2e-6 / 4e-6、更新範囲を狭める、修正済み parquet、全 epoch export、seed 固定 | **US$10〜20** / 新上限の承認が前提 | 上位 1〜9 を済ませた後でのみ解釈可能になる |

---

## 11. M4 へ進むかどうか

**進んではならない。戻るべきである。ただし「M3 をやり直す」でもない。**

1. **M4 が前提にしている control checkpoint が存在しない。**
   M3 の役割は壊れていない基準を作ることで、それが得られなかった。
2. **M4 の設計変数を正当化していた原因診断が誤りだった。**
   原因が特定できていない状態で「意図的に声側を強める」工程へ進むのは、
   変数を 1 つ足すだけで何も分離できない。
3. **判定に使った計器のうち 3 つが壊れている。**
   基準線 control、崩壊検出器、条件 4 の一貫性基準。
   直さずに M4 を走らせても、**M4 の結果も同じ理由で判定できない。**
4. **予算が超過している。** 承認を求める前に「その GPU で何を決着させるのか」を
   言えなければならない。現時点では言えない。

> **注記（2026-08-31）**: 3 の計器 3 件は M3-R 第 1 段で修復済みであり
> （[M3-R 実行計画](./j-moshi-tsukuyomi-ojousama-m3r-plan.md) 1-1〜1-5）、
> 4 の上限は 2026-08-24 に US$125 へ改定された（台帳の `cap_raise`。現在値は台帳を正とする）。
> **1 と 2 は解消していない。この節の結論は変わらない。**

---

## 12. M3 の実行そのものは擁護される

私が採用できないと言っているのは**診断**であって、**規律**ではない。

- 課金前のローカル 18 ステップで **13 件の欠陥を潰した**
- 「loss だけで判定しない」設計があったからこそ、**loss 上は成功に見えるものを不合格にできた**
- 採点段階の敵対的検証が、条件 3 の判定の誤りを 2 件捕まえた

そして今回の調査も、同じ規律の続きである。
`CLAUDE.md` の「ゲートが妥当な結果を却下し続けるならゲートを疑え」は、
**条件 4 に対して初めて適用された。**
