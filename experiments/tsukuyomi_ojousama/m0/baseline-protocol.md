# M0 Stage 2 / Stage 3 baseline再評価プロトコル

更新日: 2026-08-18

## 固定checkpoint

- Stage 2: `ayousanz/moshi-persona-stage2-ojousama-2026-07-06` revision `828b0d2b5a7e5262b137cc110d66000a2202cc39`
- Stage 3: `ayousanz/moshi-persona-stage3-ojousama-2026-07-06` revision `224b3ce8408d013cad65c16da213d2f464cc3f90`

## 音声生成

つくよみちゃんtest splitの10音声を共通promptにする。A channelへ原音、B channelへ同長の無音を置き、24 kHz stereoへ変換する。Mimi token化後、text streamはpadding、prompt 40 frames（3.2秒）、continuation 125 frames（10秒）、seed `20260818`、temperature `0.8`、top-k `0`、top-p `0`で両checkpointを評価する。

入力indexは`../eval/voice-seen-heldout-20.jsonl`の`held-out` 10件を使う。生成token、decoded WAV、実行config、stdout/stderr、checkpoint checksumを`/workspace/experiment-artifacts/baselines/{stage2,stage3}/evaluation`へ保存し、Vast.ai instance破棄前に外部へexportする。

転送対象のcanonical filenameは`heldout-prompt-files.txt`へ固定し、Vast.ai上では`run_baseline.sh`を実行する。スクリプト内のPython entrypointはすべて`uv run --no-sync`経由とする。

### prompt長を50 framesから40 framesへ変更した理由

2026-08-18に、当初の`prompt 50 frames（4秒）`がheld-out最短の`VOICEACTRESS100_026.wav`に対して余裕ゼロであることを確認した。

同ファイルは3.802秒（96 kHzで364,957 sample、24 kHzで91,239 sample）で、Mimiは`48 frames`を出力する（`prompt-dataset-report.json`の`min_frames_observed`で実測）。`utils.data.delay_and_pad_streams`が`max(delays)=1`と先頭のinitial token 1 frameを加えるため、`filter_out_short_streams`が見る長さは`48 + 1 + 1 = 50`になる。`min_length = --prompt_length = 50`との比較は`50 >= 50`で成立するので、**このファイルは50 framesでも脱落しない**。

問題は余裕が1 frameも無いことである。`filter_out_short_streams`はlogを出さずにexampleを捨てる実装なので、収録の差、resamplerの違い、delay patternの変更などで1 frame短くなるだけで、Stage 2 / Stage 3が10件ではなく9件で比較され、記録は「10件」のまま残る。held-out 10件はM1で固定した評価セットであり件数を削れないため、prompt長を`40 frames（3.2秒）`へ下げて`8 frames（0.64秒）`の余裕を確保する。過去記事の数値と直接比較しないbaselineであるため、prompt長の変更は比較可能性を損なわない。

### 無言脱落を検出するGate

`run_baseline.sh`はparquet生成後、GPUを使う前に`tools.prepare_baseline_prompts verify-dataset`を実行する。

- prompt件数が`10`と一致しない場合は失敗させる。
- いずれかのpromptが`--min-frames`（= `--prompt_length`）を下回る場合は、脱落するpromptと frame数を挙げて失敗させる。
- A/Bのframe数不一致とstream数異常を検出する。
- `generate.py`が`dialogue_id`列を削除して`example_id`を行順で振り直すため、`example_id → dialogue_id`の対応を`prompt-dataset-report.json`へ記録する。

生成後は`generated_tokens/*.npy`と`generated_wavs/*.wav`がstageごとに10件あることを確認し、一致しなければ失敗させる。

## 2026-08-18実行時に判明した未解決の2点

実行記録は`../reports/m0-baseline-run-2026-08-18.json`。prompt準備までは10件すべて通過し、checkpoint checksumも`artifact-recovery.md`と一致したが、以下2点でこのprotocolは現状のまま完了できない。どちらも本protocolの定義自体を変える判断を要する。

### 1. 生成: 公開checkpointは`dep_q=8`で`generate.py`が動かない

公開Stage 2/3の`moshi_lm_kwargs.json`は`n_q=16`、`dep_q=8`、`depformer_context=8`である。`tools/clean_moshi.py`を`--remove_modules_for_user_stream`付きで通した推論用形式で、user streamは入力としては扱うが生成しない。

一方`models/moshi_for_generation.py`の`step`は`1 + dep_q = 9` tokenを作って`num_codebooks = 1 + n_q = 17`と一致するかassertするため、`AssertionError: torch.Size([1, 9])`で停止する。このリポジトリの生成経路は`dep_q == n_q`（user streamも生成する学習形式）専用である。

選択肢:

- A: `MoshiForConditionalGeneration`をuser stream外部供給に対応させる。B channelは無音なので、prompt + generation長ぶんの無音tokenをdatasetへ入れて教師強制する。baselineの定義が「user streamは無音を教師強制」へ変わる。
- B: 公開checkpointは`moshi`公式推論スタックで動かす。`clean_moshi.py`が本来想定している経路。
- C: M0のbaselineをperplexityのみに縮小し、生成音声はM3のcontrol比較から取る。

### 2. 口調perplexity: 再実装が一様分布より悪い

Stage 2で`preferred_mean_nll = 12.88`、perplexity `391,531`、preferred勝数`1/10`、log-prob差`-64.40`となった。`text_card = 32000`の一様分布のNLLは`10.373`なので、この再実装はランダムより悪く、口調選好を測っていない。

tokenizerは原因ではない。READMEの99〜105行がJ-Moshiの`rinna/japanese-gpt2-medium` `spiece.model`使用を明記しており、実行時もそれを固定revisionで使った。

原因は入力構成にあると見られる。再実装は全codebookの音声streamを系列全体で`zero_token_id`に固定するが、`zero_token_id`は学習時に損失の`ignore_index`および系列末尾のpaddingとして使われる値であり、音声条件として入力される状態ではない。過去記事の全文と実コードは回収できておらず（`artifact-recovery.md`）、この再実装が記事の方式と一致している保証もない。

選択肢:

- A: 音声条件を`zero_token_id`固定ではなく実際のMimi tokenにする。held-out promptの音声を条件に置いて継続テキストを採点する。記事の数値とは別指標になる。
- B: 過去記事の方式の再現を諦め、口調評価をStyle held-out 50 pairの人手・選好評価に寄せる。
- C: 記事の方式を著者に確認できるまで、口調perplexityをM0の完了条件から外す。

同じTsukuyomi promptを与えても過去あみたろ音声そのものの類似性評価にはならない。このbaselineの目的は、Stage 2/3間の生成安定性、明瞭度、loop、口調差を同一条件で固定することである。

## 口調perplexity

`../eval/persona-baseline-10.jsonl`を新規10 pair `reconstructed-v1`として固定する。rinna SentencePieceを固定revisionで使用し、ブログの方式どおりaudio streamを`zero_token_id`で埋め、各continuationのtoken log probabilityを合計する。

- primary: preferred勝数 / 10、preferred合計log-prob − dispreferred合計log-prob。
- secondary: token平均NLLとperplexity（長さ差の診断用）。
- Stage 2 / Stage 3で同一pairとtokenizerを使う。

記事に掲載された過去10 pair全文は回収できないため、記事のStage 2 `+11.86`、Stage 3 `+12.26`と新しい数値は直接比較しない。計算方法の再現と、今回以後の共通baseline固定を目的とする。
