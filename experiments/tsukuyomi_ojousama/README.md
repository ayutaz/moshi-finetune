# つくよみちゃん系の声 × お嬢様口調 実験

J-Moshi-ext に、つくよみちゃんコーパス由来の声質と自然なお嬢様口調を追加学習する実験の
データ・評価セット・実行記録を置く。

進捗と完了判定の正本は [マイルストーン文書](../../docs/experiments/j-moshi-tsukuyomi-ojousama-milestones.md)、
技術方針と run matrix の正本は [実験計画](../../docs/experiments/j-moshi-tsukuyomi-ojousama-plan.md)。
このディレクトリはその根拠となる成果物を保持する。

## 状態

M0（過去baseline固定・Vast.ai基盤）と M1（権利・データ確定）は完了。M2（Tsukuyomi TTS）に着手できる。

## ディレクトリ

| パス | 内容 |
| --- | --- |
| `registry/` | データ源ごとの台帳。取得元、版、checksum、利用条件、実験で使うかどうか |
| `manifests/` | つくよみちゃんコーパス100件の行単位manifest（checksum、split、クレジット） |
| `eval/` | 固定評価セットと採点rubric。学習データからは常に除外する |
| `reports/` | 検証と実行の結果。数値の正本 |
| `m0/` | baseline再評価のprotocol、環境manifest、費用台帳、実行スクリプト |
| `reference/` | 参照専用データと、その再配布に必要なライセンス本文 |
| `style/` | お嬢様100対話の再生成仕様 |
| `DATA_CREDITS.md` | クレジット文面と非公開対象 |

## どれを見ればよいか

- **M0の結果**: `reports/m0-baseline-final.json`。生成音声20件のchecksumと、口調選好 Stage 2 / Stage 3 = 7/10
- **baselineの実行条件**: `m0/baseline-protocol.md`。prompt長やuser stream教師強制の理由も含む
- **実行環境の再現**: `m0/bootstrap_instance.sh` → `m0/run_baseline.sh`
- **費用**: `m0/spend-ledger.json`。上限 US$100
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
