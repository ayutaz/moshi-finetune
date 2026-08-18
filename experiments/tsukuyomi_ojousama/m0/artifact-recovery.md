# M0 過去artifact回収・欠損一覧

更新日: 2026-08-18

過去ブログ、Git全ref、stash/tag、到達不能object、ローカル主要保存先、Hugging Face cache、公開Hugging Face reposを調査した。元学習サーバの識別子、W&B project/run ID、外部ストレージURIは記事やリポジトリに残っておらず、対象を特定できないため未回収とした。

## 探索記録

| 探索先 | 結果 |
| --- | --- |
| Git local/remote branches | `main`、`research/j-moshi-tool-token-finetuning`、現作業branchのみ。過去voice/persona実験branchなし |
| Git tags/stashes | 該当なし |
| Git全履歴のファイル名 | 過去config、manifest、`persona_perplexity.py`なし |
| Git unreachable objects | 1 blobを確認したが別件のMoshi agent調査文書 |
| Desktop配下の主要ローカル保存先 | 過去script/config/checkpoint/audioなし |
| Hugging Face local cache | Stage 3の`refs/main`だけ存在し、weight blobなし |
| W&Bローカル認証 | 認証用netrcの存在だけ確認。project/run IDがなく、対象runを安全に限定できない |
| 過去学習サーバ・クラウドストレージ | host、instance ID、bucket/pathが記録されておらず探索不能 |
| 公開Hugging Face | Stage 2 / Stage 3を固定revisionで回収中。voice checkpointは認証対象 |

## 回収できたファイル

| Artifact | 固定版・場所 | 状態 |
| --- | --- | --- |
| Stage 2 README | HF `ayousanz/moshi-persona-stage2-ojousama-2026-07-06` revision `828b0d2b5a7e5262b137cc110d66000a2202cc39` | Vast.aiへ回収済み |
| Stage 2 config | 同上 `moshi_lm_kwargs.json`、980 bytes | Vast.aiへ回収済み |
| Stage 2 weight | 同上 `model.safetensors`、15.4 GB、SHA-256 `69a4a0112663695371a61d56372f605f549e0613e1e4f767294e2ab3811bc381` | Vast.aiへ回収済み。公開HF pointerのSHA-256と照合対象 |
| Stage 3 README | HF `ayousanz/moshi-persona-stage3-ojousama-2026-07-06` revision `224b3ce8408d013cad65c16da213d2f464cc3f90`、SHA-256 `150745041d852fe21fd94d5fa056d75df5c30332999a46219fffad11309e3282` | 回収済み |
| Stage 3 config | 同上 `moshi_lm_kwargs.json`、980 bytes、SHA-256 `2f72e3c0365cc858c23fb563d6e82b1b13a08706d6e5cb20770fff07d8ffa849` | 回収済み |
| Stage 3 weight | 同上 `model.safetensors`、15,375,500,136 bytes、SHA-256 `f34b52b7c2865cc6809e2a1c0ec527de025bdf66d3163e2c9b43ccd1d7c2c072` | 回収済み |
| 過去実験条件 | ユーザーのvoice/styleブログ2記事 | config値と結果を計画文書へ転記済み |
| Perplexity計算仕様 | styleブログの要約コード | zero audio token、continuation total log-prob方式をTDDで再実装 |

Stage 3の`.gitattributes`も回収済みだが、実験再現に直接使わない。公開repo内の原音クレジットとCC-BY-NC-4.0表記はREADMEとmodel card双方で再監査する。

## 回収できなかったファイル

| 過去記録上のファイル・成果物 | 状態と影響 | 対応 |
| --- | --- | --- |
| `configs/j-moshi-ext-amitaro.yaml` | 本体なし。ブログ抜粋のみ | 抜粋値をcontrol計画に固定し、未知値は推測せず明示する |
| `configs/accelerate_ds.yaml` / `configs/zero3-bf16-nooffload.json` | 本体なし | 現repoのDeepSpeed configとの差分を新run manifestに残す |
| `synthetic-dialogs-2026-07-04/stereo/dlg_001.wav`〜`dlg_100.wav` | 100音声すべてなし | 過去音声の再利用はしない |
| `synthetic-dialogs-2026-07-04/manifest.jsonl` | なし | ブログ記載schemaだけ回収 |
| `synthetic-dialogs-2026-07-04/dialogs_source.jsonl` | なし | 今回はつくよみちゃん用scriptを新規生成する |
| voice checkpoint `ayousanz/phase1b-jmoshi-ft-2026-07-06` | 公開取得できず、ローカルblobなし | 今回のbaseには使わず、Stage 2/3の親としてだけ記録 |
| `ojousama_mild_100/manifest.jsonl` | なし | `style/DATASET_SPEC.md`で100対話の再生成条件を固定 |
| `configs/persona-tempformer.yaml` | 本体なし。ブログ抜粋のみ | Stage 2の既知条件を記録 |
| `configs/persona-full.yaml` | 本体なし。ブログ抜粋のみ | Stage 3の既知条件を記録 |
| 過去ZeRO-3 checkpoint / optimizer state / logs | なし | 公開推論weightだけをbaselineに使用 |
| 過去`persona_perplexity.py` | 本体なし | `tools/persona_perplexity.py`としてブログ方式を再実装 |
| 過去10 pair全文 | 記事には3例と一部gain例だけで、全文なし | 新規10 pairを`reconstructed-v1`として固定し、過去の`+11.86/+12.26`と直接比較しない |
| 過去baseline生成音声 | なし | 同一prompt・seedでStage 2/3を新規生成する |
| W&B run/artifact | run ID不明 | 今後はrun IDとartifact URIを必須manifest項目にする |

## 判定

過去結果の完全再現に必要なtraining artifactは回収不能だが、公開Stage 2/3推論weight、architecture config、ブログの計算仕様は回収できた。M0 baselineは「公開weightを同一の新規固定promptで再評価すること」とし、過去の非公開10 pair数値を再現したとは表記しない。
