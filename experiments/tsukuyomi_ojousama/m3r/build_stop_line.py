"""reports/m3r-stop-line.json を組み立てる。

なぜコミットされているか
------------------------
M3-R 第2段で、dataset を作った 6 本のスクリプトが gitignore された `data/` の下にあり、
手順がリポジトリに残らなかったことが欠陥として記録された（m3r/DATASET_BUILD.md §0）。
打ち切り線の報告も同じ形で失われうるので、ここに置く。

preflight は `tools/experiment_budget.py` を **subprocess で本当に実行し**、
stdout と終了コードをそのまま記録する。ライブラリ関数を呼ぶと終了コードが記録できない。

    uv run --no-sync python experiments/tsukuyomi_ojousama/m3r/build_stop_line.py \
      --out experiments/tsukuyomi_ojousama/reports/m3r-stop-line.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

RATE = 3.0566666666666666  # m0/spend-ledger.json hourly_rate (instance 48370306)
CHEAP = 0.45  # 仮定。単一 24 GB カードの率。vastai で未確認
SPENT = 102.697  # m0/spend-ledger.json accrued_estimate.total
CAP = 125.0  # m0/spend-ledger.json experiment_cap
NEW = round(0.90 * CAP, 3)  # tools/experiment_budget.NEW_RUN_FRACTION
STEP_SECONDS = 85.94  # reports/m3-v0-training.json seconds_per_step
STEPS = 45
EXPORT_BYTES = 16.74e9  # reports/m3-instance-bootstrap.json train-dq16 weights_gb
EXPORT_BYTES_PER_SECOND = 8.5556e6  # m3-report §8: 15.4 GB in about 30 minutes
EXPORT_COUNT = 5
CONTINGENCY_HOURS = 0.5

PREFLIGHTS = [
    (
        "phase4-as-planned-6.0h",
        ["--spent", "102.697", "--hourly-rate", str(RATE), "--planned-hours", "6.0"],
    ),
    (
        "phase4-bottom-up-6.24h",
        ["--spent", "102.697", "--hourly-rate", str(RATE), "--planned-hours", "6.24"],
    ),
    (
        "phase4-with-contingency-7.5h",
        ["--spent", "102.697", "--hourly-rate", str(RATE), "--planned-hours", "7.5"],
    ),
    (
        "phase4-training-half-3.0h",
        ["--spent", "102.697", "--hourly-rate", str(RATE), "--planned-hours", "3.0"],
    ),
    (
        "phase4-max-that-passes-3.20h",
        ["--spent", "102.697", "--hourly-rate", str(RATE), "--planned-hours", "3.20"],
    ),
    (
        "phase4-just-over-3.21h",
        ["--spent", "102.697", "--hourly-rate", str(RATE), "--planned-hours", "3.21"],
    ),
    (
        "forward-only-4-1-0.5h-cheap",
        ["--spent", "102.697", "--hourly-rate", "0.40", "--planned-hours", "0.5"],
    ),
    (
        "cheap-box-after-training-half",
        ["--spent", "111.867", "--hourly-rate", "0.45", "--planned-hours", "4.0"],
    ),
    (
        "cheap-box-alone-4.0h",
        ["--spent", "102.697", "--hourly-rate", "0.45", "--planned-hours", "4.0"],
    ),
    (
        "cheap-box-max-21.7h",
        ["--spent", "102.697", "--hourly-rate", "0.45", "--planned-hours", "21.7"],
    ),
    (
        "split-run1-big-2.9h",
        ["--spent", "102.697", "--hourly-rate", str(RATE), "--planned-hours", "2.9"],
    ),
    (
        "split-run2-cheap-5.2h-cap125",
        ["--spent", "111.487", "--hourly-rate", "0.45", "--planned-hours", "5.2"],
    ),
    (
        "split-run2-cheap-5.2h-cap127",
        ["--spent", "111.487", "--hourly-rate", "0.45", "--planned-hours", "5.2", "--cap", "127"],
    ),
    (
        "single-box-7.05h-cap125",
        ["--spent", "102.697", "--hourly-rate", str(RATE), "--planned-hours", "7.05"],
    ),
    (
        "single-box-7.05h-cap139",
        [
            "--spent",
            "102.697",
            "--hourly-rate",
            str(RATE),
            "--planned-hours",
            "7.05",
            "--cap",
            "139",
        ],
    ),
]

STAGES = [
    (
        "rent + bootstrap（j-moshi-ext snapshot 約 15.76 GB の DL を含む）",
        1.38 / RATE,
        "plan",
        "m3-plan step 19 の US$1.38 ÷ 3.0567。M3 でも単独では計測していない。"
        "DL 量は m0/spend-ledger.json の m3-voice-control-rental preflight（Bootstrap pulls 15.76 GB）。"
        ".claude/skills/vast-run/SKILL.md の 31 GB は M0 の 2 checkpoint 版で、"
        "m3/bootstrap_m3_instance.sh は j-moshi-ext 1 本しか引かない",
    ),
    (
        "base model dq16 / dq8 を構築し、parquet を upload",
        1.07 / RATE,
        "plan",
        "m3-plan step 20 の US$1.07。同上",
    ),
    ("smoke test と disk 検算", 0.92 / RATE, "plan", "m3-plan step 21 の US$0.92。同上"),
    (
        "control 生成 50 prompt × 2 sample",
        2 * 0.76 / RATE,
        "plan",
        "m3-plan step 22 の US$0.76 × 2（m3r-plan が prompt あたり複数 sample を要求する）",
    ),
    (
        "学習 45 step",
        STEPS * STEP_SECONDS / 3600,
        "measured",
        "reports/m3-v0-training.json: seconds_per_step 85.94、wall_clock 1 h 04 m。"
        "M3-R の行は 244.57 frame で M3 の 261.45 より 6.5% 短いが、短い側では見積もらない",
    ),
    (
        "ZeRO state 5 本を bf16 へ変換",
        0.50,
        "plan-split",
        "m3-plan step 24 の US$3.21 が変換と生成を合算している。ここで割った内訳であり、単独の計測はない",
    ),
    ("生成 5 epoch × 50 prompt × 2 sample", 2 * (3.21 / RATE - 0.50), "plan-split", "同上"),
    (
        "export 5 本 × 16.74 GB を手元へ転送",
        EXPORT_COUNT * EXPORT_BYTES / EXPORT_BYTES_PER_SECOND / 3600,
        "measured",
        "m3-report §8: 15.4 GB あたり約 30 分 = 8.56 MB/s。"
        "16.74 GB は reports/m3-instance-bootstrap.json の train-dq16 weights_gb",
    ),
    ("destroy と台帳精算", 0.15 / RATE, "plan", "m3-plan step 30 の US$0.15"),
]

SPLIT_RUN1 = [
    ("rent + bootstrap", 1.38 / RATE),
    ("base model + upload", 1.07 / RATE),
    ("smoke test", 0.92 / RATE),
    ("学習 45 step", STEPS * STEP_SECONDS / 3600),
    ("ZeRO state 5 本を変換", 0.50),
    ("84 GB を保管インスタンスへ push", 0.15),
    ("destroy と台帳精算", 0.15 / RATE),
]
SPLIT_RUN2 = [
    ("bootstrap", 0.30),
    ("4-1 forward で base loss 内訳", 0.50),
    ("control 生成 ×2", 2 * 0.76 / RATE),
    ("生成 5 epoch ×2", 2 * (3.21 / RATE - 0.50)),
    ("84 GB の転送元になる", EXPORT_COUNT * EXPORT_BYTES / EXPORT_BYTES_PER_SECOND / 3600),
    ("destroy", 0.05),
]
EGRESS_USD = 0.22  # 84 GB。台帳の 34.368 GB -> US$0.090 から

EXAMPLE_START = dt.datetime(2026, 8, 28, 9, 0, tzinfo=dt.UTC)


def hm(hours: float) -> str:
    minutes = int(round(hours * 60))
    return f"{minutes // 60}h{minutes % 60:02d}m"


def iso(moment: dt.datetime) -> str:
    return moment.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def run_preflights() -> dict:
    out = {}
    for label, args in PREFLIGHTS:
        finished = subprocess.run(
            [sys.executable, "-m", "tools.experiment_budget", *args],
            capture_output=True,
            text=True,
            check=False,
        )
        out[label] = {
            "command": "uv run --no-sync python -m tools.experiment_budget " + " ".join(args),
            "exit_code": finished.returncode,
            "stdout": json.loads(finished.stdout),
        }
    return out


def cumulative(items):
    rows, total = [], 0.0
    for name, hours in items:
        total += hours
        rows.append(
            {
                "stage": name,
                "hours": round(hours, 3),
                "cumulative_hours": round(total, 3),
                "deadline_offset": hm(total),
            }
        )
    return rows, round(total, 3)


def build_core() -> dict:
    rows, total = [], 0.0
    for name, hours, basis, source in STAGES:
        total += hours
        rows.append(
            {
                "stage": name,
                "hours": round(hours, 3),
                "usd": round(hours * RATE, 3),
                "cumulative_hours": round(total, 3),
                "deadline_offset": hm(total),
                "basis": basis,
                "source": source,
            }
        )
    run1_rows, run1_hours = cumulative(SPLIT_RUN1)
    run2_rows, run2_hours = cumulative(SPLIT_RUN2)
    run1_usd, run2_usd = round(run1_hours * RATE, 3), round(run2_hours * CHEAP, 3)
    split_total = round(run1_usd + run2_usd + EGRESS_USD, 3)
    budget_hours = (NEW - SPENT) / RATE
    return {
        "rate_usd_per_hour": RATE,
        "cheap_rate_assumed": CHEAP,
        "accrued": SPENT,
        "cap": CAP,
        "new_run_limit": NEW,
        "headroom_usd": round(NEW - SPENT, 3),
        "headroom_hours_at_rate": round(budget_hours, 3),
        "estimate_rows": rows,
        "estimate_total_hours": round(total, 3),
        "estimate_total_usd": round(total * RATE, 3),
        "estimate_predicted_cumulative": round(SPENT + total * RATE, 3),
        "estimate_required_cap": round((SPENT + total * RATE) / 0.90, 2),
        "split_run1_rows": run1_rows,
        "split_run1_hours": run1_hours,
        "split_run1_usd": run1_usd,
        "split_run2_rows": run2_rows,
        "split_run2_hours": run2_hours,
        "split_run2_usd": run2_usd,
        "split_egress_usd": EGRESS_USD,
        "split_total_usd": split_total,
        "split_predicted_cumulative": round(SPENT + split_total, 3),
        "split_required_cap": round((SPENT + split_total) / 0.90, 2),
        "preflights": run_preflights(),
        "worked_example": {
            "if_start_date_reads": iso(EXAMPLE_START),
            "budget_line": iso(EXAMPLE_START + dt.timedelta(hours=budget_hours)),
            "budget_line_offset": f"+{budget_hours:.3f} h（({NEW} − {SPENT}) ÷ {RATE:.4f}）",
            "work_line_single_box_plan": iso(
                EXAMPLE_START + dt.timedelta(hours=total + CONTINGENCY_HOURS)
            ),
            "work_line_single_box_offset": f"+{total + CONTINGENCY_HOURS:.3f} h（見積もり {total:.3f} + 予備 {CONTINGENCY_HOURS}）",
            "work_line_split_run1": iso(
                EXAMPLE_START + dt.timedelta(hours=run1_hours + CONTINGENCY_HOURS)
            ),
            "work_line_split_run1_offset": f"+{run1_hours + CONTINGENCY_HOURS:.3f} h（見積もり {run1_hours:.3f} + 予備 {CONTINGENCY_HOURS}）",
            "stop_line_single_box_plan": iso(
                EXAMPLE_START + dt.timedelta(hours=min(total + CONTINGENCY_HOURS, budget_hours))
            ),
            "stop_line_split_run1": iso(
                EXAMPLE_START
                + dt.timedelta(hours=min(run1_hours + CONTINGENCY_HOURS, budget_hours))
            ),
            "what_the_example_shows": "単箱の計画では budget line が work line より 4.33 時間早い。打ち切り線は仕事の終わりではなく"
            "金の終わりで決まり、仕事はそこまでに終わらない。分割計画の run1 でも budget line がなお 10 分早く、"
            "0.5 時間の予備は使えない。予備を積む余地は予算の側にないというのはこの意味である。",
            "note": "この 2 つは run ごとに計算し直す。上の値は形式を示すためのもので、次の run の値ではない。",
        },
    }


core = build_core()


READ_START_DATE = (
    'vastai show instance $INSTANCE --raw | python3 -c "\n'
    "import datetime as dt, json, os, sys\n"
    "row = json.load(sys.stdin)\n"
    "start = dt.datetime.fromtimestamp(row['start_date'], dt.timezone.utc).replace(microsecond=0)\n"
    "rate = float(os.environ['RATE']); spent = float(os.environ['SPENT'])\n"
    "limit = float(os.environ['NEW_RUN_LIMIT']); planned = float(os.environ['PLANNED_HOURS'])\n"
    "work = (start + dt.timedelta(hours=planned)).replace(microsecond=0)\n"
    "budget = (start + dt.timedelta(hours=(limit - spent) / rate)).replace(microsecond=0)\n"
    "print('start_date  ', start.isoformat())\n"
    "print('work line   ', work.isoformat())\n"
    "print('budget line ', budget.isoformat())\n"
    "print('STOP LINE   ', min(work, budget).isoformat(), '<- 早い方')\n"
    '" | tee m3r-stop-line.txt'
)

WATCH = (
    'vastai show instance $INSTANCE --raw | python3 -c "\n'
    "import datetime as dt, json, os, sys\n"
    "row = json.load(sys.stdin)\n"
    "start = dt.datetime.fromtimestamp(row['start_date'], dt.timezone.utc).replace(microsecond=0)\n"
    "now = dt.datetime.now(dt.timezone.utc).replace(microsecond=0)\n"
    "elapsed = (now - start).total_seconds() / 3600\n"
    "rate = float(os.environ['RATE']); spent = float(os.environ['SPENT'])\n"
    "print('start_date', start.isoformat())\n"
    "print('now       ', now.isoformat())\n"
    "print('elapsed   ', round(elapsed, 2), 'h')\n"
    "print('accrued   ', round(spent + elapsed * rate, 3))\n"
    "open('.m3r-accrued', 'w').write(str(round(spent + elapsed * rate, 3)))\n"
    '"\n'
    "uv run --no-sync python -m tools.experiment_budget \\\n"
    '  --spent "$(cat .m3r-accrued)" --hourly-rate "$RATE" --planned-hours "$REMAINING_HOURS"\n'
    'echo "budget preflight exit=$?"'
)

report = {
    "schema_version": 1,
    "milestone": "M3-R",
    "step": "3-4 打ち切り線の確定",
    "captured_at": "2026-08-27",
    "applies_to": "docs/experiments/j-moshi-tsukuyomi-ojousama-m3r-plan.md 第4段（4-1 〜 4-5）",
    "verdict": "第4段は、いまの予算では現行の計画のまま開始できない。見積もり 7.041 時間 US$21.52 に対し、"
    "preflight が許す残余は US$9.803 = 3.207 時間しかない。打ち切り線を引く前に、"
    "予算の判断が要る。",
    "why_the_line_comes_before_the_work": (
        "M3 は 14.0 時間の線に対し 25.21 時間動き、超過分 US$34.27 がそのまま上限突破の原因になった。"
        "そのとき線を守れなかった理由 3 つは m0/spend-ledger.json の cap_breach.why_it_was_not_caught に"
        "記録されている。この文書は、その 3 つそれぞれに置き換えを与える。"
    ),
    "rate": {
        "usd_per_hour": core["rate_usd_per_hour"],
        "source": "m0/spend-ledger.json hourly_rate。M3 のインスタンス 48370306（2x A100-SXM4-80GB, 900 GB disk）の dph_total。",
        "caveat": "次に借りるインスタンスの率ではない。レンタル時に dph_total を読み、この文書の全数値を引き直すこと。",
    },
    "estimate": {
        "method": "M3 で実測された 2 つの値と、M3 計画のステップ別 US$ 見積もりから積み上げる。"
        "実測値と計画値を混ぜたので、どちらかを行ごとに basis 欄に書いた。",
        "what_is_actually_measured": [
            "学習 1 step あたり 85.94 秒（reports/m3-v0-training.json）",
            "checkpoint の転送 15.4 GB あたり約 30 分 = 8.56 MB/s（docs/.../m3-report.md §8）",
        ],
        "what_is_not_measured": (
            "bootstrap・base model 構築・smoke test・変換・生成の各所要時間は、M3 でも単独では計測されていない。"
            "M3 セッションは 25.21 時間で、計画の 6.60 時間の 3.82 倍だったが、"
            "**その 18.6 時間の超過をステップに割り振る記録がない。** したがって計画値を出典とする行は、"
            "3.8 倍まで外れうる値である。"
        ),
        "the_first_thing_the_4th_stage_must_do": (
            "各ステップの開始・終了を UTC で記録すること。M3 のセッションは事後に分解できず、"
            "そのせいでこの見積もりの 6 行が計画値のままになっている。"
        ),
        "sequence_length_note": "M3-R の train 行は 244.57 frame、M3 は 261.45 frame で 6.5% 短い。"
        "線形なら 45 step が 1.074 → 1.005 時間になるが、短い側では見積もらない。",
        "stages": core["estimate_rows"],
        "total_hours": core["estimate_total_hours"],
        "total_usd": core["estimate_total_usd"],
        "plus_4_1_forward_measurement_usd": 0.40,
        "predicted_cumulative_usd": core["estimate_predicted_cumulative"],
    },
    "budget": {
        "accrued_estimate": core["accrued"],
        "accrued_source": "m0/spend-ledger.json accrued_estimate.total（invoiced_to_date ではない）",
        "cap": core["cap"],
        "new_run_limit": core["new_run_limit"],
        "new_run_limit_is_what_binds": (
            "上限 US$125 に対する残余は US$22.30 だが、preflight が判定するのは上限の 90%、US$112.50 である。"
            "実際に到達できる残余は US$9.803 しかない。cap_raise が「V-real 1 腕 US$18 + forward US$0.40」を"
            "買ったと書いているが、その US$18.40 は preflight を通らない。"
        ),
        "headroom_usd": core["headroom_usd"],
        "headroom_hours_at_rate": core["headroom_hours_at_rate"],
        "breakpoint_measured": {
            "allow_at_hours": 3.20,
            "reject_at_hours": 3.21,
            "evidence": "preflights.phase4-max-that-passes-3.20h（exit 0）と phase4-just-over-3.21h（exit 1）",
        },
        "splitting_the_run_does_not_help": (
            "予測は spent + rate × hours であり、spent は run をまたいで積み上がる。"
            "3.0 時間の run を 1 本走らせると spent が US$111.867 になり、以後どんな長さの run も"
            "予測が US$112.50 を超える。実測: cheap-box-after-training-half が exit 1。"
            "分割で買えるのは最初の 3.2 時間だけである。"
        ),
        "required_cap": {
            "for_the_plan_as_estimated": core["estimate_required_cap"],
            "for_the_restructured_split": core["split_required_cap"],
            "note": "required cap = 予測累計 ÷ 0.90。tools/experiment_budget.py の NEW_RUN_FRACTION による。",
        },
    },
    "preflights": {
        "what_this_is": "実行した preflight とその終了コードをそのまま記録したもの。緩めていない。",
        "runs": core["preflights"],
    },
    "stop_line": {
        "rule": "打ち切り線は経過時間ではなく UTC 時刻である。レンタル直後に一度だけ計算し、"
        "ファイルに書き出して、以後は再計算しない。",
        "definition": "STOP = min(work_line, budget_line)。"
        "work_line = start_date + 見積もり時間 + 0.5 時間の予備。"
        "budget_line = start_date + (new_run_limit − accrued) ÷ dph_total。",
        "start_date_is_the_authority": (
            "M3 の進捗確認は毎回、計画のステップ見積もりから予測を作り直していた。"
            "予測は安心なままで、時計は違った。読むのは instance の start_date だけである。"
        ),
        "how_to_compute_it_once": {
            "environment": {
                "INSTANCE": "<vastai instance id>",
                "RATE": "vastai show instance --raw の dph_total",
                "SPENT": "m0/spend-ledger.json accrued_estimate.total",
                "NEW_RUN_LIMIT": "m0/spend-ledger.json new_run_prediction_limit",
                "PLANNED_HOURS": "この文書の見積もり + 0.5",
            },
            "command": READ_START_DATE,
        },
        "worked_example": core["worked_example"],
        "stage_deadlines": {
            "what_they_are": "start_date からの累積オフセット。レンタル時に UTC の絶対時刻へ直して"
            "m3r-stop-line.txt に並べる。各行を過ぎてもそのステップが終わっていなければ、"
            "次へ進まずに export して止める。",
            "single_box_plan": core["estimate_rows"],
            "split_plan_run1": core["split_run1_rows"],
            "split_plan_run2": core["split_run2_rows"],
        },
        "the_export_is_not_the_end_of_the_run": (
            "M3 は最後の生成完了から転送完了までの間、経過時間を一度も確認していない。"
            "5 本 × 16.74 GB の転送は 2.718 時間で、見積もり全体の 39% を占める。"
            "「学習が終わった」は run の中間点であって終点ではない。"
        ),
    },
    "during_the_run": {
        "cause_3_it_replaces": "tools/experiment_budget.py をレンタル前にしか呼ばなかった",
        "every_minutes": 20,
        "why_20": "1 回 20 分は US$1.02。見落としの上限をそこに固定する。"
        "M3 は 11.21 時間 US$34.27 を見落とした。",
        "environment": {
            "INSTANCE": "<id>",
            "RATE": "dph_total",
            "SPENT": "m0/spend-ledger.json accrued_estimate.total（run 開始時点の値。更新しない）",
            "REMAINING_HOURS": "この時点から終わりまでに要ると見ている時間",
        },
        "command": WATCH,
        "read_all_four_lines": [
            "start_date — 毎回読む。頭の中の経過時間は使わない",
            "elapsed — 打ち切り線と比べる",
            "accrued — spent + elapsed × rate",
            "budget preflight exit — 0 以外なら、いま持っているものを export して止める",
        ],
        "what_stops_the_run": [
            "preflight が 0 以外を返した。ステップの途中でも止める。ステップこそが金の行き先である",
            "現在時刻が STOP を過ぎた",
            "起動 assertion が Num examples / batch / steps と違う（下の launch_assertion を見よ）",
            "転送が始まって 40 分たっても 1 本目が終わらない（8.56 MB/s の想定より遅い。残り時間を引き直す）",
        ],
        "before_stopping": [
            "export を先に済ませる。disk は stop では残るが destroy で消える",
            "destroy 前に sha256 を突き合わせる",
            "900 GB を停止状態で置くと US$10.00/日。終わったインスタンスは当日中に destroy する",
        ],
    },
    "launch_assertion": {
        "why_it_is_here": "3-3 の一致検査が、reports/m3r-tokenize.json が出荷前の 72 行 parquet を"
        "記録したままであることを見つけた。m3r-plan 4-2 のゲートは「Num examples 72」で、"
        "出荷 parquet では trainer が 70 と印字する。**正しい run が古い台帳を根拠に kill される。**",
        "correct_values": {
            "Num examples": 70,
            "Total train batch size": 8,
            "Total optimization steps": 45,
        },
        "measured_from": "出荷 parquet の実測行数 70（reports/m3r-dataset-agreement.json parquet.train.rows）",
        "steps_are_unchanged": "ceil(70/8) = 9、9 × 5 = 45。M3 と同じ 45 step である。",
        "action_if_different": "kill immediately",
        "fix_before_the_4th_stage": "reports/m3r-tokenize.json と m3r-plan 4-2 のゲート文言を直すこと。",
    },
    "options_to_make_it_fit": [
        {
            "option": "A. 上限を上げてもらう",
            "what": f"現行計画のまま走らせるには上限 US${core['estimate_required_cap']} が要る。",
            "evidence": "preflights.single-box-7.05h-cap125（exit 1）と single-box-7.05h-cap139（exit 0）",
            "who_decides": "利用者。上限は利用者の決定であり、こちらが再承認するものではない"
            "（m0/spend-ledger.json cap_breach.before_any_further_gpu_work）。",
        },
        {
            "option": "B. export を高い機械から外す",
            "what": "見積もりの 2.718 時間 US$8.31 は、手元へ 84 GB を落とす転送であり、"
            "律速は手元の回線である。それを US$3.0567/時の機械で待っている。"
            "変換まで終えたら 5 本を安いインスタンスへ push し、高い方を destroy してから落とす。",
            "run1_hours": core["split_run1_hours"],
            "run1_usd": core["split_run1_usd"],
            "run2_hours": core["split_run2_hours"],
            "run2_usd": core["split_run2_usd"],
            "egress_usd": core["split_egress_usd"],
            "total_usd": core["split_total_usd"],
            "predicted_cumulative": core["split_predicted_cumulative"],
            "required_cap": core["split_required_cap"],
            "what_it_buys": f"必要な上限が US${core['estimate_required_cap']} から US${core['split_required_cap']} に下がる。"
            "run1 単独なら現行の上限で preflight を通る（split-run1-big-2.9h が exit 0）。",
            "evidence": "preflights.split-run1-big-2.9h（exit 0）、split-run2-cheap-5.2h-cap125（exit 1）、"
            "split-run2-cheap-5.2h-cap127（exit 0）",
            "not_verified": "安いインスタンスの率 US$0.45/h、24 GB カードで dq16 の生成が載ること、"
            "インスタンス間の転送が速いこと。いずれも vastai search offers で確かめられるが、"
            "このステップでは vastai を実行していない。",
        },
        {
            "option": "C. export する epoch を減らす",
            "what": "5 本を 3 本にすると 1.087 時間 US$3.32 減る。",
            "why_not": "全 epoch の export は、M3 が両 arm の最良である epoch 2 を失ったことへの対策そのものである"
            "（m3r-plan 第4段の設定表）。減らすと、この計画が買おうとしたものを削ることになる。"
            "しかも US$3.32 では足りない。",
        },
        {
            "option": "D. 何もしない",
            "what": "第4段を開始しない。第0〜3段の成果（訂正済みの記録、直った計器、作り直した dataset、"
            "ローカルの測定）は残り、課金はゼロのままである。",
            "cost": "control checkpoint が得られないので M4 は Blocked のまま。",
        },
    ],
    "the_three_causes_and_what_replaces_them": [
        {
            "cause": "進捗確認のたびに計画の見積もりから予測を再計算し、instance の start_date を読まなかった",
            "replacement": "stop_line.how_to_compute_it_once で線を一度だけ UTC 時刻に固定し、"
            "during_the_run.command が毎回 start_date を読む。予測は再計算しない。",
        },
        {
            "cause": "checkpoint の転送が想定よりはるかに遅く、最後の生成完了から転送完了まで経過時間を一度も確認しなかった",
            "replacement": "転送を見積もりの 1 行として計上した（2.718 時間、全体の 39%）。"
            "stage_deadlines に転送の締切がある。40 分で 1 本目が終わらなければ残りを引き直す。",
        },
        {
            "cause": "tools/experiment_budget.py をレンタル前にしか呼ばなかった",
            "replacement": "during_the_run が 20 分ごとに呼ぶ。0 以外の終了コードは、ステップの途中でも止める合図である。",
        },
    ],
    "records_that_disagree_with_the_evidence": [
        {
            "where": ".claude/skills/vast-run/SKILL.md",
            "says": "bootstrap re-downloads 31 GB of checkpoints",
            "evidence": "experiments/tsukuyomi_ojousama/m3/bootstrap_m3_instance.sh は j-moshi-ext の "
            "snapshot 1 本だけを引く。m0/spend-ledger.json の m3-voice-control-rental preflight は "
            "Bootstrap pulls 15.76 GB と記録している。31 GB は M0 の 2 checkpoint 版である。",
            "effect": "見積もりには効かない（この行の時間は US$ 見積もりから来ている）。"
            "offer 選びで inet_down を過大に要求する。",
        },
        {
            "where": ".claude/skills/vast-run/SKILL.md",
            "says": "tools/experiment_budget.py still hardcodes HARD_CAP = 100.0 ... "
            "the preflight refuses every run - its non-zero exit is evidence of nothing either way",
            "evidence": "tools/experiment_budget.py は DEFAULT_HARD_CAP = 125.0 で、閾値は上限の "
            "0.75 / 0.90 / 0.95 の割合になっている。m0/spend-ledger.json threshold_note が"
            "同じ修正を記録している。",
            "effect": "**危険な残存である。** この段落は「preflight の非ゼロ終了は何の証拠でもない」と読める。"
            "いまは違う。3.21 時間で返る reject-new-run は本物であり、無視すれば M3 と同じことが起きる。"
            "この文書の preflights はすべて修正後の道具で取ったものである。",
        },
    ],
    "limits": [
        "見積もりの 9 行のうち 6 行は M3 計画の US$ 見積もりが出典で、実測ではない。M3 セッションは計画の 3.82 倍かかった。"
        "その差をステップに割り振る記録がないので、この 6 行の誤差幅は測れていない。",
        "0.5 時間の予備は小さい。大きくしなかったのは安心のためではなく、3.21 時間で preflight が落ちるからである。"
        "予備を積む余地は予算の側にない。",
        "安いインスタンスの率と能力は仮定である。vastai を実行していない。",
        "16.74 GB は base model dq16 の重みで、fine-tune 後の export サイズは M3 では計測されていない"
        "（M3 が checksum まで確認したのは dq8 control の 15,375,500,136 バイト）。",
    ],
}

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--out", help="ここへ書く。省略すると標準出力")
args = parser.parse_args()
text = json.dumps(report, ensure_ascii=False, indent=2)
if args.out:
    Path(args.out).write_text(text + "\n", encoding="utf-8")
    print(f"wrote {args.out}")
else:
    print(text)
