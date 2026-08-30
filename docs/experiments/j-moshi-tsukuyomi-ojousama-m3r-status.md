# M3-R 現在地 — 何が終わり、何で止まり、次に何を決めるか

作成日: 2026-08-28

**M3-R は第0〜3段と 4-1 を通過し、4-2（V-real 学習）の run1 で止まっている。**
run1 は学習に到達しないまま US$4.289 を使って終わり、**現行の上限では再挑戦の preflight が
通らない。** 次に GPU を動かすには上限の判断が要る。

この文書は現在地の一枚要約である。手順とゲートは
[M3-R 実行計画](./j-moshi-tsukuyomi-ojousama-m3r-plan.md)、完了判定は
[マイルストーン](./j-moshi-tsukuyomi-ojousama-milestones.md)、金額は
[`m0/spend-ledger.json`](../../experiments/tsukuyomi_ojousama/m0/spend-ledger.json)、
run1 の詳細は
[`reports/m3r-run1-failure.json`](../../experiments/tsukuyomi_ojousama/reports/m3r-run1-failure.json)
が正本である。**この文書と出典が食い違えば出典が正。**

---

## 1. 終わっているもの

| 段 | 内容 | 課金 | 証拠 |
| --- | --- | ---: | --- |
| 第0段 | 撤回された原因診断を記録から除去 | US$0 | [M3 検証記録](./j-moshi-tsukuyomi-ojousama-m3-verification.md) |
| 第1段 | 計器を直す（崩壊検出器に音響指標、条件4を区間推定へ、eval loss 内訳、seed 必須化） | US$0 | `reports/m3-collapse-acoustic.json`、`reports/m3-likeness-calibration.json` |
| 第2段 | dataset `v-real-v2` を出荷（train 70 / dev 8、総 step 45、裸 `▁` 0.0000、重なり 1,202 frame） | US$0 | `reports/m3r-timeline.json`、`reports/m3r-tokenize.json`、`reports/m3r-roomtone.json` |
| 第3段 | ローカルで測り切る（round-trip、明瞭度、一致検査、UTC 打ち切り線） | US$0 | `reports/m3r-roundtrip.json`、`reports/m3r-intelligibility.json`、[`m3r/STOP_LINE.md`](../../experiments/tsukuyomi_ojousama/m3r/STOP_LINE.md) |
| 第4段 4-1 | base loss の内訳を forward 1 回で実測 | **US$0.115** | [`reports/m3r-forward-breakdown.json`](../../experiments/tsukuyomi_ojousama/reports/m3r-forward-breakdown.json) |

表中の `reports/…` と `m3r/…` は
[`experiments/tsukuyomi_ojousama/`](../../experiments/tsukuyomi_ojousama/) の下にある。

4-1 が確定させたのは、**base audio loss の 80.1% は `models/utils.py` が deepcopy した未学習の
複製ヘッドが占め、話者A側は合わせて 17.0%（semantic 15.7% + acoustic 1.3%）にすぎない**という
内訳である。M3 の base audio loss を「対象話者を予測できない証拠」として読むことはできない。

第0〜3段のうち未達のまま残るゲートは 1 件（`pad ⟺ 話者A無音` の一致率 60% 未満。0.907 → 0.809 まで）。
それが代理していた「loss を下げる最短経路が無音」の方は、話者A静音フレームの
デジタル無音率 87.05% → 0 で断ってある。

## 2. 止まっている理由 — run1 は学習に到達しなかった

2026-08-27、instance 48911872（A100 80GB **PCIe** ×2）。

| | |
| --- | --- |
| 停止位置 | `Generating train split: 70 examples` の直後。`finetune.py:820-827` の `with accelerator.main_process_first():` を抜ける地点 |
| 症状 | 両ランクが CPU 100%、utime は増え続ける（10 秒で 10 秒分）。45 秒間、HF datasets キャッシュにも `/workspace` にも書き込みなし |
| どこまで来ていたか | NCCL は init 済み。**起動 assertion（`Num examples 70` / batch 8 / steps 45）に一度も届いていない** |
| 課金 | **US$4.289**（1.72 h × US$2.4936/h）。打ち切り線 3.376 h の手前で自主中断 |
| 成果 | **ゼロ。** checkpoint も loss も出ていない |

自主中断の理由は、残り 1.78 h に対し学習 1.07 + 変換 0.5 = 1.57 h で、猶予 0.21 h では
単一 GPU への切り替えを試して失敗した場合に変換へ届かないためである。

**否定した仮説 3 件**（いずれも run1 の原因ではない）

| 仮説 | 検証 | 結果 |
| --- | --- | --- |
| `num_proc=16` の並列 map | `--dataset_processing_workers 1` で再起動 | 同じ地点で 19 分停止。**否定** |
| `preprocess_function` が重い | 出荷 parquet 70 行を**手元で**通した（課金ゼロ） | 0.01 秒で完了、70 例・shape (17,280)。**否定** |
| `futex_wait_queue` = デッドロック | 実処理ランクの utime を 10 秒間隔で比較 | 増加していた。見ていたのは親プロセスの値。**誤診** |

## 3. 予算の現在地 — 再挑戦を弾いているのはここ

| | |
| --- | ---: |
| 累計 `accrued_estimate` | **US$107.301** |
| 上限 | US$125.00 |
| 上限までの残余 | US$17.699 |
| **preflight の限度**（`new_run_prediction_limit`） | **US$112.50** |
| **preflight が許す残余** | **US$5.199** |
| 走行中の停止線 | US$118.75 |
| 日次課金 | **US$0.00**（この実験のインスタンスはすべて破棄済み） |

| 何に | インスタンス | 課金 |
| --- | --- | ---: |
| M0・M2（baseline 生成、TTS、停止中 disk 2 日ぶん） | 48004205 / 48178589 / 48187958 | 25.638 |
| M3 本走（25.21 h。打ち切り線 14.0 h を 11.21 h 超過） | 48370306 | 77.059 |
| M3-R 4-1（forward、0.382 h） | 48838452 | 0.115 |
| M3-R 4-2 の選定やり直し（走らせずに破棄） | 48911444 | 0.200 |
| M3-R 4-2 run1（1.72 h、成果ゼロ） | 48911872 | 4.289 |
| **計** | | **107.301** |

**preflight の実測**（`uv run --no-sync python -m tools.experiment_budget`、緩めていない）

| 対象 | `--spent` | `--hourly-rate` | `--planned-hours` | 予測 | status | exit |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| run1 の再挑戦（run1 の実績率） | 107.301 | 2.4936 | 3.376 | **115.719** | **reject-new-run** | **1** |
| 同（M3 の SXM4 率で引くと） | 107.301 | 3.0567 | 2.876 | **116.092** | **reject-new-run** | **1** |
| bootstrap から smoke test まで | 107.301 | 2.4936 | 1.103 | 110.051 | allow-with-warning | 0 |
| 生成・転送の箱（Tesla P40） | 107.301 | 0.1067 | 4.76 | 107.809 | allow-with-warning | 0 |

**拘束するのは上限 US$125 ではなく、その 90% の US$112.50 である。**

## 4. 未検証の仮説 — NCCL の P2P が PCIe で成立していない

| | GPU | 結果 |
| --- | --- | --- |
| M3（成功） | A100-**SXM4**-80GB（NVLink） | 45 step 完走 |
| M3-R run1（失敗） | A100 80GB **PCIe** | collective でハング |

`vastai search offers` の結果はどちらも「A100」としか表示しない。**GPU 種別の選定ミスである。**
NCCL は init を抜けており、両ランクの CPU 100% と utime 増加は busy-wait と整合する。
**ただしこれは未検証で、GPU を借りないと確かめられない。**

確かめ方は、次に借りたときの最初の数分に置く。

```bash
NCCL_P2P_DISABLE=1 NCCL_DEBUG=INFO accelerate launch --use_deepspeed ... --max_train_steps 2
```

報告書はこの 2 step を **US$0.5** と見積もっている（run1 のように起動済みの箱で、本番の代わりに
2 step だけ回した場合の費用である）。
新しく借りる場合は bootstrap と base model 構築が先に要るので、smoke test に届くまでは
計画の a〜c = 1.103 時間、run1 の率で **US$2.75** である。**この範囲だけなら現行の上限で通る**（§3）。

## 5. 利用者が決める必要があること

**上限の判断は利用者のものである。以下は選択肢と必要額であって、推奨ではない。**

**そして、止めているのはこれだけである。** §4 の NCCL 仮説は選択肢 (1) の範囲で潰せるが、潰しても
4-2 の完走は上限の内側に入らない。§6 に残る未決2件はいずれも 4-2 の結果待ちで、先には決められない。

| 選択肢 | 必要な上限 | できること | できないこと |
| --- | ---: | --- | --- |
| (1) 据え置き | US$125 | 診断のみ（bootstrap + 2 step smoke test、US$2.75 前後）。第0〜3段と 4-1 の成果は残る | run1 の完走（予測 115.719 > 112.50、exit 1）。M4 は Blocked のまま |
| (2) run1 を 1 本通す | **US$128.58 以上** | run1（学習 45 step + bf16 変換、3.376 h） | US$128.58 ちょうどでは限度との差が US$0.003 しかなく、率が少しでも上がれば再び reject する |
| (3) 残りの工程まで通す | **US$129.15 以上**（US$130 で確認） | run1 に加えて生成・転送（Tesla P40 US$0.1067/h × 4.76 h = US$0.508）。run1 後の予測 116.227 が US$130 の限度 US$117.00 の内側 | — |
| (4) 第4段を開始しない | US$125 | 第0〜3段と 4-1 の成果（計器・dataset・base loss 内訳）は残る | control checkpoint が得られず、M4 は Blocked のまま |

**前提**: 率 US$2.4936/h は run1 の実績（A100 PCIe ×2、disk 500 GB）であり、SXM4 を名指しすれば
変わる。`dph_total` に disk 代は含まれない（`rate = dph_total + storage_cost × disk_gb / 730`）。
**借りる直前に率を取り直し、preflight を通し直すこと。**

## 6. 未決事項の現在

| # | 事項 | 状態 |
| --- | --- | --- |
| 未決1 | acoustic 損失重みを振るか | **control 待ちのまま。** 4-1 が「A 側 acoustic の寄与は audio_total の 1.3%」を確定させたが、`--acoustic_loss_weight` は A 側と複製ヘッドを分けられない（`finetune.py:689-712`）。条件4 の `closed` を見るまで、重みが第一容疑者か M4 の変数かを決められない |
| 未決2 | warmup を何 step にするか | **決着。** `--num_warmup_steps 0` を明示する（DeepSpeed が `max(2, ·)` にクランプして実効 2 step）。候補 0 件の時点で事前登録された値で、M3 と同一である |
| 未決3 | 生成をどの箱で走らせるか | 価格の面では選択肢 (b)（20 GiB 級 1 本で生成と転送を兼ねる）が成立。所要時間は M3 計画由来の見積もりのままで、**run1 の実測で置き換える予定だった。run1 が走っていないので未確定** |

## 7. 次に借りるときに変えること

commit `ae14898` で [`vast-run` skill](../../.claude/skills/vast-run/SKILL.md) に反映済み。

- **SXM4 を名指しする。** 検索条件に `gpu_name=A100_SXM4` を入れる。検索結果の「A100」は
  PCIe と SXM4 を区別しない
- **2 step の smoke test を本番の前に置く。** M3 の計画にはこの段があり、run1 は飛ばした
- **CPU だけで完結する段は借りる前に手元で通す。** `preprocess_function` は 0.01 秒だった
- **率は `dph_total + storage_cost × disk_gb / 730` で計算する。** これを怠った 48911444 は
  走らせずに破棄して US$0.20 の授業料になった

**文書に書くだけでは守られなかったので hook にした**（commit `918249a`）。
配線と運用は [M3-R 実行計画](./j-moshi-tsukuyomi-ojousama-m3r-plan.md) の「自動で効くもの」が正本である。

- `vastai create instance` の直前に**実請求率と interconnect** が出る（警告のみ、止めない）
- セッション終了時に**経過時間と打ち切り線との差**が出る。線は借りた直後に
  [`m0/spend-ledger.json`](../../experiments/tsukuyomi_ojousama/m0/spend-ledger.json) の
  `active_stop_lines` へ書く（現在は空）
- **smoke test を飛ばすことだけは hook で守れない。** 発火点が学習コマンドを打つ瞬間だからである
