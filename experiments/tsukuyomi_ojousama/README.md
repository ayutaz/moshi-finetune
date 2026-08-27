# つくよみちゃん系の声 × お嬢様口調 実験

J-Moshi-ext に、つくよみちゃんコーパス由来の声質と自然なお嬢様口調を追加学習する実験の
データ・評価セット・実行記録を置く。

進捗と完了判定の正本は [マイルストーン文書](../../docs/experiments/j-moshi-tsukuyomi-ojousama-milestones.md)、
技術方針と run matrix の正本は [実験計画](../../docs/experiments/j-moshi-tsukuyomi-ojousama-plan.md)。
このディレクトリはその根拠となる成果物を保持する。

## 状態

M0（過去baseline固定・Vast.ai基盤）、M1（権利・データ確定）、M2（Tsukuyomi TTS）は完了。
**M3（Voice control）は完了（不合格）。不合格の判定は正しかったが、原因の読みは撤回済みである**
（[M3実施報告](../../docs/experiments/j-moshi-tsukuyomi-ojousama-m3-report.md) と、
撤回の根拠を持つ [M3検証記録](../../docs/experiments/j-moshi-tsukuyomi-ojousama-m3-verification.md)）。

いま進んでいるのは **M3-R（Voice control 再走）** で、記録・計器・データを直したうえで、
V-real 1腕で壊れていないcontrolを取り直す工程である。実行順序とゲートは
[M3-R実行計画](../../docs/experiments/j-moshi-tsukuyomi-ojousama-m3r-plan.md)を正とする。
M4は渡せるcontrol checkpointがないためBlockedのまま。

| 段 | 内容 | 状態 | 証拠 |
| --- | --- | --- | --- |
| 第0段 | 撤回された原因診断を記録から除去する | 完了 | [M3検証記録](../../docs/experiments/j-moshi-tsukuyomi-ojousama-m3-verification.md) |
| 第1段 | 計器を直す（崩壊検出器に音響指標、条件4を区間推定へ、eval loss、seed必須化） | 完了 | `reports/m3-collapse-acoustic.json`、`reports/m3-likeness-calibration.json`、`reports/m3-text-stream-audit.json` |
| 第2段 | datasetを作り直す（読点分割・相槌・重なり・ルームトーン。**連結は撤回**） | 完了 | `reports/m3r-tokenize.json`、`reports/m3r-timeline.json`、`reports/m3r-roomtone.json` |
| 第3段 | ローカルで測り切る（投入tokenのround-trip、明瞭度、一致検査、打ち切り線） | 完了 | `reports/m3r-roundtrip.json`、`reports/m3r-intelligibility.json`、`reports/m3r-dataset-agreement.json`、[`m3r/STOP_LINE.md`](m3r/STOP_LINE.md) |
| 第4段 4-1 | base lossの内訳をforward 1回で実測（US$0.115、instance破棄済み） | 完了 | `reports/m3r-forward-breakdown.json` |
| 第4段 4-2以降 | V-real学習 → 全epoch変換・export → 生成 → インスタンス破棄 | 未着手 | — |

**GPU予算の上限はUS$125**（2026-08-24承認。M3セッションがUS$102.697で旧上限US$100を
突破したあとの引き上げ）。累計は **US$102.812**。拘束力を持つのは上限そのものではなく
new_run_limit `US$112.50` の方である。台帳は `m0/spend-ledger.json`。

### 出荷したdataset（`v-real-v2`）

| | M3-R | M3 |
| --- | ---: | ---: |
| train / dev 行数 | 70 / 8 | 72 / 8 |
| 総step（`ceil(70/8)×5`） | 45 | 45 |
| turn / 対話 | 4.95 | 3.00 |
| 同時発話フレーム | 1,202 | 0 |
| 裸 `▁` が出力text tokenに占める割合 | 0.0000 | 0.455 |
| 最短フレーム（train / dev） | 201 / 202 | — |

200フレームの床を割ったv-047とv-057は出荷から外した（床は下げない）。判断の記録は
`reports/m3r-timeline.json` の `min_frames_floor.decision`。行数が落ちても総stepは45のまま、
M3とも過去の成功runとも構成が揃う。

### 4-1で分かったこと

base audio lossの **80.1% は、`models/utils.py` がuser stream用にdeepcopyした未学習ヘッド**が
占めていた。話者A側は17.0%（semantic 2.57 = chanceの33.7%、accuracy 37.6%）にすぎない。
**M3のbase audio loss 6.82〜7.19を「j-moshi-extが対象話者を予測できない証拠」として
読むことはできない。** 内訳は `reports/m3r-forward-breakdown.json`。

## ディレクトリ

| パス | 内容 |
| --- | --- |
| `registry/` | データ源ごとの台帳。取得元、版、checksum、利用条件、実験で使うかどうか |
| `manifests/` | つくよみちゃんコーパス100件の行単位manifest（checksum、split、クレジット）と、対話datasetのmanifest・tokenize sidecar |
| `eval/` | 固定評価セットと採点rubric。学習データからは常に除外する |
| `reports/` | 検証と実行の結果。数値の正本 |
| `m0/` | baseline再評価のprotocol、環境manifest、費用台帳、実行スクリプト |
| `m2/` | Tsukuyomi TTSインスタンスのbootstrap |
| `m3/` | 対話datasetの仕様（`DATASET_SPEC.md`）と、学習インスタンスのbootstrap |
| `m3r/` | M3-Rのdataset構築手順、tokenizeコマンド、一致検査、第4段の打ち切り線 |
| `reference/` | 参照専用データと、その再配布に必要なライセンス本文 |
| `style/` | お嬢様100対話の再生成仕様 |
| `DATA_CREDITS.md` | クレジット文面と非公開対象 |

## どれを見ればよいか

- **M0の結果**: `reports/m0-baseline-final.json`。生成音声20件のchecksumと、口調選好 Stage 2 / Stage 3 = 7/10
- **M2の結果**: `reports/m2-tts-gate.json`（客観ゲート）、`reports/m2-listening-judgement.json`（聴取判定）。
  採用は T1 speaker inversion（学習パラメータ12,288、base完全凍結）
- **M2の再現条件**: `reports/m2-run-manifest.json`。base weightのSHA-256、config、seed、ローカルパッチまで含む
- **M3の不合格判定**: `reports/m3-voice-control-gate.json`。原因診断の撤回は
  [M3検証記録](../../docs/experiments/j-moshi-tsukuyomi-ojousama-m3-verification.md)
- **M3-Rの前提測定**: `reports/m3-speaker-b-probe.json`（話者Bは1本凍結でないと一人にならない）、
  `reports/m3-local-compute-probe.json`（Mimiはローカルで動き、160対話が1.2分）、
  [データセット監査](../../docs/experiments/j-moshi-tsukuyomi-ojousama-m3r-dataset-audit.md)（連結の撤回）
- **M3-Rの計器**: `tools/dialogue_collapse.py`（text行だけでなくaudio行を見る）、
  `tools/likeness_guard.py`（条件3と条件4のインターロック）、
  `tools/intelligibility.py`（perplexityと反復を必ず並べて出す）
- **M3-Rのdataset構築**: [`m3r/DATASET_BUILD.md`](m3r/DATASET_BUILD.md)、
  [`m3r/TOKENIZE_COMMANDS.md`](m3r/TOKENIZE_COMMANDS.md)（[`m3r/PENDING_CORRECTIONS.md`](m3r/PENDING_CORRECTIONS.md) の2件が未適用）、
  仕様は [`m3/DATASET_SPEC.md`](m3/DATASET_SPEC.md)
- **第4段の打ち切り線と起動assertion**: [`m3r/STOP_LINE.md`](m3r/STOP_LINE.md)。
  起動時に印字されるべき値は `Num examples 70` / `Total train batch size 8` / `Total optimization steps 45`
- **baselineの実行条件**: `m0/baseline-protocol.md`。prompt長やuser stream教師強制の理由も含む
- **実行環境の再現**: `m0/bootstrap_instance.sh` → `m0/run_baseline.sh`、TTSは `m2/bootstrap_tts_instance.sh`、
  学習インスタンスは `m3/bootstrap_m3_instance.sh`（2x A100 80GBと空きRAM 80GiBに満たなければ起動時に止まる）
- **費用**: `m0/spend-ledger.json`。上限 US$125
- **データの由来と権利**: `registry/` と `DATA_CREDITS.md`

`reports/m0-baseline-run-2026-08-18.json` と `-2026-08-20.json` は失敗記録として残してある。
`m0-baseline-final.json` の `supersedes` が関係を示す。

## 非公開のもの

つくよみちゃんコーパスの原音、生成した音声、公開条件を確認する前の派生checkpointは
リポジトリに含めない。`data/` 配下（gitignore対象）に置く。詳細は `DATA_CREDITS.md`。

`reference/ojousama-talk-script-201.jsonl` はMITライセンスの派生物としてリポジトリに含めており、
`reference/LICENSE.OjousamaTalkScriptDataset` の許諾条項本文と併せて配布する必要がある。

## 検証

これらの成果物の整合は `tests/test_experiment_assets.py` が守る。

```bash
uv run --python 3.12 --with pytest --no-sync python -m pytest tests -q
```

原音を必要とする検査は、データが無い環境では自動でskipする。
