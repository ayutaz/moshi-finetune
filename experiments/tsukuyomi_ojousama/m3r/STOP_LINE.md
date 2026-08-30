# M3-R 第4段の打ち切り線 — 借りる前に決め、走っている間に読むもの

作成日: 2026-08-27
正本の数値: [`reports/m3r-stop-line.json`](../reports/m3r-stop-line.json)
組み立て: `uv run --no-sync python experiments/tsukuyomi_ojousama/m3r/build_stop_line.py --out experiments/tsukuyomi_ojousama/reports/m3r-stop-line.json`
（この文書の表はすべてそこから来る。食い違ったら JSON が正）

> **⚠ 2026-08-31: 予算の数値は run1 の前のものである。**
> この文書は accrued **US$102.812** の時点で書かれた。run1（4-2）が
> **US$4.489** を使って失敗した結果、現在は次のとおりである。
>
> | | この文書 | **現在** |
> | --- | ---: | ---: |
> | accrued | 102.812 | **107.301** |
> | preflight が許す残余 | US$9.803 | **US$5.199** |
> | 買える時間（US$3.0567/h） | 3.207 h | **1.70 h** |
>
> **打ち切り線の引き方（`STOP = min(work_line, budget_line)`、`start_date` を読む、
> 走行中も preflight を呼ぶ）は変わっていない。金額だけが変わった。**
> 正は [`m0/spend-ledger.json`](../m0/spend-ledger.json) の `accrued_estimate.total`。
> 借りる前に `build_stop_line.py` を現在の accrued で回し直すこと。
> 経緯は [M3-R 現在地](../../../docs/experiments/j-moshi-tsukuyomi-ojousama-m3r-status.md)。

---

## 0. 結論を先に

**第4段は、いまの予算では現行の計画のまま開始できない。**

| | |
| --- | ---: |
| 見積もり | **7.041 時間 / US$21.52** |
| preflight が許す残余 | **US$9.803 = 3.207 時間** |
| 差 | 3.83 時間ぶん足りない |

上限 US$125 に対する残余は US$22.30 ある。しかし `tools/experiment_budget.py` が判定するのは
**上限の 90%、US$112.50** である。`cap_raise` が「V-real 1 腕 US$18 + forward US$0.40」を
買ったと書いているが、**その US$18.40 は preflight を通らない。**

打ち切り線を引く作業は済ませた（§3〜§5）。使えるのは、§6 の判断が済んでからである。

---

## 1. 時間の見積もり

M3 で**実測された値は 2 つだけ**である。残りは M3 計画の US$ 見積もりを時間に直したもので、
**M3 セッションはその計画の 3.82 倍かかった**（25.21 時間対 6.60 時間）。
その 18.6 時間の超過をステップに割り振る記録は残っていない。

| # | ステップ | 時間 | US$ | 出典 |
| --- | --- | ---: | ---: | --- |
| 1 | rent + bootstrap（j-moshi-ext snapshot 約 15.76 GB の DL 込み） | 0.451 | 1.380 | 計画 |
| 2 | base model dq16 / dq8 構築、parquet upload | 0.350 | 1.070 | 計画 |
| 3 | smoke test と disk 検算 | 0.301 | 0.920 | 計画 |
| 4 | control 生成 50 prompt × 2 sample | 0.497 | 1.520 | 計画 |
| 5 | **学習 45 step** | **1.074** | 3.284 | **実測** `reports/m3-v0-training.json` 85.94 s/step |
| 6 | ZeRO state 5 本を bf16 へ変換 | 0.500 | 1.528 | 計画（変換と生成の合算を割ったもの） |
| 7 | 生成 5 epoch × 50 prompt × 2 sample | 1.100 | 3.363 | 同上 |
| 8 | **export 5 本 × 16.74 GB を手元へ転送** | **2.718** | 8.307 | **実測** m3-report §8 の 15.4 GB / 30 分 = 8.56 MB/s |
| 9 | destroy と台帳精算 | 0.049 | 0.150 | 計画 |
| | **計** | **7.041** | **21.522** | |

**転送だけで全体の 39% である。** M3 が「最後の生成完了から転送完了まで経過時間を一度も
確認しなかった」のは、この 2.7 時間を見積もりに入れていなかったからである。

系列長は M3-R のほうが 6.5% 短い（244.57 対 261.45 frame）ので、線形なら学習は 1.005 時間に
なる。**短い側では見積もらない。**

> **第4段が最初にやること: 各ステップの開始と終了を UTC で記録する。**
> M3 のセッションは事後に分解できず、そのせいでこの表の 6 行が計画値のままである。

---

## 2. 予算 preflight — 実行結果をそのまま

緩めていない。`--cap` を既定のまま呼んだものが本体である。

```bash
uv run --no-sync python -m tools.experiment_budget \
  --spent 102.697 --hourly-rate 3.0566666666666666 --planned-hours <H>
```

| H | status | 予測累計 | exit |
| ---: | --- | ---: | ---: |
| 6.00（計画） | `reject-new-run` | 121.037 | **1** |
| 6.24 | `reject-new-run` | 121.771 | **1** |
| 7.05（この文書の見積もり） | `reject-new-run` | 124.247 | **1** |
| 7.50（予備込み） | `reject-new-run` | 125.622 | **1** |
| 3.21 | `reject-new-run` | 112.509 | **1** |
| **3.20** | `allow-with-warning` | 112.478 | **0** |
| 3.00 | `allow-with-warning` | 111.867 | 0 |

**境目は 3.20 時間と 3.21 時間の間**にある。US$9.803 ÷ US$3.0567/h = 3.207 時間。

### 分割しても伸びない

予測は `spent + rate × hours` で、`spent` は run をまたいで積み上がる。

| | spent | H | status | exit |
| --- | ---: | ---: | --- | ---: |
| run 1（3.0 時間） | 102.697 | 3.0 | `allow-with-warning` | 0 |
| run 2（run 1 のあと） | 111.867 | 4.0 @ US$0.45/h | `reject-new-run` | **1** |

**3.0 時間の run を 1 本走らせると、以後どんな長さの run も通らない。**
分割で買えるのは最初の 3.2 時間だけである。

---

## 3. 打ち切り線 — UTC 時刻で

**経過時間ではない。** レンタル直後に一度だけ計算し、ファイルに書き出す。以後は再計算しない。

```
STOP        = min(work_line, budget_line)
work_line   = start_date + 見積もり時間 + 0.5 時間の予備
budget_line = start_date + (new_run_prediction_limit − accrued_estimate) ÷ dph_total
```

`start_date` が権威である。M3 の進捗確認は毎回、計画のステップ見積もりから予測を作り直していた。
**予測は安心なままで、時計は違った。**

```bash
INSTANCE=<vastai instance id>
RATE=<vastai show instance --raw の dph_total>
SPENT=<m0/spend-ledger.json accrued_estimate.total>
NEW_RUN_LIMIT=<m0/spend-ledger.json new_run_prediction_limit>
PLANNED_HOURS=<§1 の見積もり + 0.5>

vastai show instance $INSTANCE --raw | python3 -c "
import datetime as dt, json, os, sys
row = json.load(sys.stdin)
start = dt.datetime.fromtimestamp(row['start_date'], dt.timezone.utc).replace(microsecond=0)
rate = float(os.environ['RATE']); spent = float(os.environ['SPENT'])
limit = float(os.environ['NEW_RUN_LIMIT']); planned = float(os.environ['PLANNED_HOURS'])
work = (start + dt.timedelta(hours=planned)).replace(microsecond=0)
budget = (start + dt.timedelta(hours=(limit - spent) / rate)).replace(microsecond=0)
print('start_date  ', start.isoformat())
print('work line   ', work.isoformat())
print('budget line ', budget.isoformat())
print('STOP LINE   ', min(work, budget).isoformat(), '<- 早い方')
" | tee m3r-stop-line.txt
```

### 例（形式を示すためのもの。次の run の値ではない）

`start_date` が `2026-08-28T09:00:00Z` なら:

| | 時刻 | オフセット |
| --- | --- | --- |
| budget line | `2026-08-28T12:12:25Z` | +3.207 h |
| work line（単箱の計画） | `2026-08-28T16:32:27Z` | +7.541 h |
| work line（分割 run 1） | `2026-08-28T12:22:33Z` | +3.376 h |
| **STOP（どちらの計画でも）** | **`2026-08-28T12:12:25Z`** | budget line が binding |

単箱の計画では budget line が work line より **4.33 時間早い**。
打ち切り線は仕事の終わりではなく**金の終わり**で決まり、仕事はそこまでに終わらない。
分割の run 1 でも budget line がなお 10 分早く、**0.5 時間の予備は使えない。**

### ステップごとの締切

`start_date` からの累積オフセット。レンタル時に UTC の絶対時刻へ直して `m3r-stop-line.txt` に
並べる。**その行を過ぎてそのステップが終わっていなければ、次へ進まずに export して止める。**

| ステップ | 累積 | オフセット |
| --- | ---: | --- |
| bootstrap 完了 | 0.451 | +0h27m |
| base model + upload 完了 | 0.802 | +0h48m |
| smoke test 通過 | 1.103 | +1h06m |
| control 生成 完了 | 1.600 | +1h36m |
| **学習 45 step 完了** | 2.674 | **+2h40m** |
| 変換 5 本 完了 | 3.174 | +3h10m |
| 生成 5 epoch 完了 | 4.274 | +4h16m |
| **転送 5 本 完了** | 6.992 | **+7h00m** |
| destroy | 7.041 | +7h02m |

---

## 4. 走っている間に呼ぶ手順

**20 分おき。** 1 回 20 分は US$1.02 で、見落としの上限をそこに固定する。
M3 は 11.21 時間 US$34.27 を見落とした。

```bash
INSTANCE=<id>
RATE=<dph_total>
SPENT=<m0/spend-ledger.json accrued_estimate.total。run 開始時点の値。更新しない>
REMAINING_HOURS=<いまから終わりまでに要ると見ている時間>

vastai show instance $INSTANCE --raw | python3 -c "
import datetime as dt, json, os, sys
row = json.load(sys.stdin)
start = dt.datetime.fromtimestamp(row['start_date'], dt.timezone.utc).replace(microsecond=0)
now = dt.datetime.now(dt.timezone.utc).replace(microsecond=0)
elapsed = (now - start).total_seconds() / 3600
rate = float(os.environ['RATE']); spent = float(os.environ['SPENT'])
print('start_date', start.isoformat())
print('now       ', now.isoformat())
print('elapsed   ', round(elapsed, 2), 'h')
print('accrued   ', round(spent + elapsed * rate, 3))
open('.m3r-accrued', 'w').write(str(round(spent + elapsed * rate, 3)))
"
uv run --no-sync python -m tools.experiment_budget \
  --spent "$(cat .m3r-accrued)" --hourly-rate "$RATE" --planned-hours "$REMAINING_HOURS"
echo "budget preflight exit=$?"
```

**4 行とも読むこと。**

| 行 | 何を見るか |
| --- | --- |
| `start_date` | 毎回読む。頭の中の経過時間は使わない |
| `elapsed` | `m3r-stop-line.txt` の STOP と比べる |
| `accrued` | `spent + elapsed × rate` |
| `budget preflight exit` | **0 以外なら、いま持っているものを export して止める** |

### 止める条件

- preflight が 0 以外を返した。**ステップの途中でも止める。ステップこそが金の行き先である**
- 現在時刻が STOP を過ぎた
- 起動 assertion が下の値と違う
- 転送が始まって 40 分たっても 1 本目が終わらない（8.56 MB/s の想定より遅い。残りを引き直す）

### 止める前に

- export を先に済ませる。disk は stop では残るが destroy で消える
- destroy 前に sha256 を突き合わせる
- 900 GB を停止状態で置くと **US$10.00/日**。終わったインスタンスは当日中に destroy する

---

## 5. 起動 assertion — 値が変わっている

3-3 の一致検査が、`reports/m3r-tokenize.json` が**出荷前の 72 行 parquet を記録したまま**で
あることを見つけた。m3r-plan 4-2 のゲートは「起動 assertion が `Num examples 72`」で、
`m3r-tokenize.json` は「違えば即 kill」と書いている。
**出荷 parquet では trainer は 70 と印字する。正しい run が、古い台帳を根拠に kill される。**

| | 正しい値 |
| --- | ---: |
| `Num examples` | **70** |
| `Total train batch size` | 8 |
| `Total optimization steps` | 45 |

`ceil(70/8) = 9`、`9 × 5 = 45`。**総 step は 72 行のときと同じ 45 で、M3 とも揃う。**
違うのは行数だけである。第4段に入る前に `reports/m3r-tokenize.json` と m3r-plan 4-2 の
文言を直すこと。根拠は [`reports/m3r-dataset-agreement.json`](../reports/m3r-dataset-agreement.json)
の `mismatches_found[1]` と `correct_launch_assertion`。

---

## 6. どう分割すれば通るか

### A. 上限を上げてもらう

現行計画のまま走らせるには**上限 US$138.02** が要る（予測累計 124.247 ÷ 0.90）。
実測: `--cap 125` で exit 1、`--cap 139` で exit 0。

**上限は利用者の決定であり、こちらが再承認するものではない**
（`m0/spend-ledger.json` `cap_breach.before_any_further_gpu_work`）。

### B. export を高い機械から外す ← 推奨

見積もりの **2.718 時間 US$8.31 は、手元へ 84 GB を落とす転送**である。
律速は手元の回線（8.56 MB/s）であって、インスタンスではない。
それを **US$3.0567/時の機械で待っている。**

変換まで終えたら 5 本を安いインスタンスへ push し、高い方を destroy してから落とす。

| | 時間 | US$ |
| --- | ---: | ---: |
| run 1（2× A100、bootstrap → 学習 → 変換 → push → destroy） | 2.876 | 8.791 |
| run 2（単一 24 GB、4-1 forward → 生成 → 転送元） | 5.165 | 2.324 |
| network egress 約 84 GB | | 0.220 |
| **計** | | **11.335** |

- 予測累計 **US$114.03**、必要な上限 **US$126.70**
- **run 1 だけなら現行の上限で通る**（`--planned-hours 2.9` で exit 0、予測 111.561）
- run 2 は `--cap 125` で exit 1、`--cap 127` で exit 0

**必要な上限が US$138.02 から US$126.70 に下がる。**

未検証の仮定（`vastai` はこのステップでは実行していない）:
安いインスタンスの率 US$0.45/h、24 GB カードで dq16 の生成が載ること、
インスタンス間の転送が速いこと。**借りる前に `vastai search offers` で確かめること。**

### C. export する epoch を減らす — 採らない

5 本を 3 本にすると 1.087 時間 US$3.32 減る。しかし全 epoch の export は
**M3 が両 arm の最良である epoch 2 を失ったことへの対策そのもの**である。
減らすと、この計画が買おうとしたものを削ることになる。しかも US$3.32 では足りない。

### D. 第4段を開始しない

第0〜3段の成果（訂正済みの記録、直った計器、作り直した dataset、ローカルの測定）は残り、
課金はゼロのままである。代償は、control checkpoint が得られず M4 が Blocked のままになること。

---

## 7. M3 の 3 つの原因と、その置き換え

| M3 の原因 | 置き換え |
| --- | --- |
| 進捗確認のたびに計画の見積もりから予測を再計算し、`start_date` を読まなかった | §3 で線を一度だけ UTC 時刻に固定。§4 のコマンドが毎回 `start_date` を読む。予測は再計算しない |
| checkpoint の転送が想定よりはるかに遅く、最後の生成完了から転送完了まで経過時間を一度も確認しなかった | 転送を見積もりの 1 行として計上（2.718 時間、全体の 39%）。ステップ締切に転送の行がある。40 分で 1 本目が終わらなければ残りを引き直す |
| `tools/experiment_budget.py` をレンタル前にしか呼ばなかった | §4 が 20 分ごとに呼ぶ。0 以外の終了コードは、ステップの途中でも止める合図 |

---

## 8. 証拠と食い違っている記録

`.claude/skills/vast-run/SKILL.md` に、実態と合わない記述が 2 件残っている。
**このステップの担当ファイルではないので直していない。**

| 記述 | 実態 | 影響 |
| --- | --- | --- |
| 「bootstrap re-downloads 31 GB of checkpoints」 | `bootstrap_m3_instance.sh` は j-moshi-ext の snapshot 1 本だけを引く。台帳の preflight は **15.76 GB** と記録している。31 GB は M0 の 2 checkpoint 版 | 見積もりには効かない。offer 選びで `inet_down` を過大に要求する |
| 「`tools/experiment_budget.py` still hardcodes `HARD_CAP = 100.0` ... its non-zero exit is **evidence of nothing either way**」 | 現在の `DEFAULT_HARD_CAP` は **125.0** で、閾値は上限の 0.75 / 0.90 / 0.95 の割合。`m0/spend-ledger.json` の `threshold_note` が同じ修正を記録している | **危険。** この段落は「preflight の非ゼロ終了は何の証拠でもない」と読める。いまは違う。3.21 時間で返る `reject-new-run` は本物である |

§2 の判定はすべて**修正後の**道具で取ったものである。

---

## 9. この文書が答えられないこと

- 見積もりの 9 行のうち **6 行は M3 計画の見積もりが出典で、実測ではない。**
  M3 セッションは計画の 3.82 倍かかったが、その差をステップに割り振る記録がないので、
  この 6 行の誤差幅は測れていない。
- **0.5 時間の予備は小さい。** 大きくしなかったのは安心のためではなく、3.21 時間で preflight が
  落ちるからである。予備を積む余地は予算の側にない。
- 安いインスタンスの率と能力は仮定である。`vastai` を実行していない。
- 16.74 GB は base model dq16 の重みで、**fine-tune 後の export サイズは M3 では計測されていない**
  （M3 が checksum まで確認したのは dq8 control の 15,375,500,136 バイト）。
