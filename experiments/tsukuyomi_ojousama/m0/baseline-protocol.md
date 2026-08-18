# M0 Stage 2 / Stage 3 baseline再評価プロトコル

更新日: 2026-08-18

## 固定checkpoint

- Stage 2: `ayousanz/moshi-persona-stage2-ojousama-2026-07-06` revision `828b0d2b5a7e5262b137cc110d66000a2202cc39`
- Stage 3: `ayousanz/moshi-persona-stage3-ojousama-2026-07-06` revision `224b3ce8408d013cad65c16da213d2f464cc3f90`

## 音声生成

つくよみちゃんtest splitの10音声を共通promptにする。A channelへ原音、B channelへ同長の無音を置き、24 kHz stereoへ変換する。Mimi token化後、text streamはpadding、prompt 50 frames（4秒）、continuation 125 frames（10秒）、seed `20260818`、temperature `0.8`、top-k `0`、top-p `0`で両checkpointを評価する。

入力indexは`../eval/voice-seen-heldout-20.jsonl`の`held-out` 10件を使う。生成token、decoded WAV、実行config、stdout/stderr、checkpoint checksumを`/workspace/experiment-artifacts/baselines/{stage2,stage3}/evaluation`へ保存し、Vast.ai instance破棄前に外部へexportする。

転送対象のcanonical filenameは`heldout-prompt-files.txt`へ固定し、Vast.ai上では`run_baseline.sh`を実行する。スクリプト内のPython entrypointはすべて`uv run --no-sync`経由とする。

同じTsukuyomi promptを与えても過去あみたろ音声そのものの類似性評価にはならない。このbaselineの目的は、Stage 2/3間の生成安定性、明瞭度、loop、口調差を同一条件で固定することである。

## 口調perplexity

`../eval/persona-baseline-10.jsonl`を新規10 pair `reconstructed-v1`として固定する。rinna SentencePieceを固定revisionで使用し、ブログの方式どおりaudio streamを`zero_token_id`で埋め、各continuationのtoken log probabilityを合計する。

- primary: preferred勝数 / 10、preferred合計log-prob − dispreferred合計log-prob。
- secondary: token平均NLLとperplexity（長さ差の診断用）。
- Stage 2 / Stage 3で同一pairとtokenizerを使う。

記事に掲載された過去10 pair全文は回収できないため、記事のStage 2 `+11.86`、Stage 3 `+12.26`と新しい数値は直接比較しない。計算方法の再現と、今回以後の共通baseline固定を目的とする。
