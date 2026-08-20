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

**採用した解決策（2026-08-20、ユーザー承認）: user streamを無音で教師強制する。**

- `MoshiForConditionalGeneration`に`num_user_codebooks`（`num_audio_codebooks - dep_q`）を追加し、depformerが生成しないcodebookを外部から受け取る。`dep_q == n_q`のcheckpointでは`0`になり従来どおり動く。
- `generate.py`はdelay適用済みの`batch.input_ids`から該当stream（行`1 + dep_q`以降）の`prompt_length`〜`prompt_length + generation_length`を切り出し、1 frameずつ供給する。frameが足りないexampleは明示的に失敗させる。
- prompt音声は`prompt_length + generation_length + 2` frameぶんまで無音でpaddingする（`+2`はdelay pattern分）。A channelは先頭に原音、以降は無音になる。prompt区間は先頭40 framesで全件が原音のみなので、paddingはpromptの内容を変えない。

したがって本baselineは「**user streamは無音を教師強制し、Moshi側のstreamだけを生成する**」条件と定義する。B channelは元から無音指定なので、protocolの意図とは矛盾しない。

### 2. 口調perplexity: 再実装が一様分布より悪い

Stage 2で`preferred_mean_nll = 12.88`、perplexity `391,531`、preferred勝数`1/10`、log-prob差`-64.40`となった。`text_card = 32000`の一様分布のNLLは`10.373`なので、この再実装はランダムより悪く、口調選好を測っていない。

tokenizerは原因ではない。READMEの99〜105行がJ-Moshiの`rinna/japanese-gpt2-medium` `spiece.model`使用を明記しており、実行時もそれを固定revisionで使った。

原因は入力構成にあると見られる。再実装は全codebookの音声streamを系列全体で`zero_token_id`に固定するが、`zero_token_id`は学習時に損失の`ignore_index`および系列末尾のpaddingとして使われる値であり、音声条件として入力される状態ではない。過去記事の全文と実コードは回収できておらず（`artifact-recovery.md`）、この再実装が記事の方式と一致している保証もない。

**実施した対応（2026-08-20、ユーザー承認）: 実音声を条件にして採点する。→ 効果なし。**

音声条件を`zero_token_id`固定から、held-out promptの実Mimi token（`VOICEACTRESS100_032`、A 8 + B 8 = 16 codebook）へ変更し、`delays`を適用して`utils.data.delay_and_pad_streams`と同じ整列にした。あわせて一様分布ゲートを追加し、`preferred_mean_nll`が`log(text_card)`を下回らないrunは診断用reportを残したうえで失敗するようにした。

結果は次のとおりで、指標は成立しなかった。

| 条件 | Stage 2 `preferred_mean_nll` | Stage 3 | 一様分布の上界 |
| --- | ---: | ---: | ---: |
| 全codebookを`zero_token_id`（2026-08-18） | `12.878` | 未実行 | `10.373` |
| 実Mimi token条件（2026-08-20） | `13.004` | `15.204` | `10.373` |

音声条件を変えてもほぼ動かないため、**音声側は原因ではない**ことが確定した。

### 確定した原因: 採点しているtext streamの形が学習時と違う

`tools/tokenize_text.py`の`tokenize_and_pad_text`は次を行う。

1. `token_ids = [text_padding_id] * num_frames`で**全frameをpaddingで初期化**する。
2. 単語のタイムスタンプから求めたframeにだけtokenを書き込む。
3. tokenの直前frameがpaddingだった場合、そこへ`end_of_text_padding_id`を挿入する。
4. tokenizeは`encode_as_pieces_wo_byte_fallback` + `piece_to_id`で、日本語では`--no_whitespace_before_word`を使う。

対して`tools/persona_perplexity.py`は、`SentencePieceProcessor.encode`で得た連続したtext tokenを密に並べて採点している。padding frameが無い、`end_of_text_padding_id`が無い、時刻整列が無い、という3点で学習分布から外れており、これがNLLが一様分布より悪くなる理由である。

### 残る判断

指標を成立させるには、completion tokenを妥当なframe位置へ配置し、間をpaddingで埋め、`end_of_text_padding_id`を入れる必要がある。これは発話タイミングのモデル化を伴う指標の再設計であり、M5のStyle Gateの定義にも影響する。次のいずれかをユーザーが選ぶまで、口調perplexityはM0の完了条件から外して保留する。

- A: 時刻整列を模した採点へ再設計する。
- B: 口調評価をStyle held-out 50 pairのブラインド選好評価と語尾分布へ寄せ、自動perplexityは使わない。
- C: 過去記事の方式を著者に確認できるまで保留する。

生成側のbaselineは固定済みなので、この保留はM2以降の進行を止めない。

同じTsukuyomi promptを与えても過去あみたろ音声そのものの類似性評価にはならない。このbaselineの目的は、Stage 2/3間の生成安定性、明瞭度、loop、口調差を同一条件で固定することである。

## 口調perplexity

`../eval/persona-baseline-10.jsonl`を新規10 pair `reconstructed-v1`として固定する。rinna SentencePieceを固定revisionで使用し、ブログの方式どおりaudio streamを`zero_token_id`で埋め、各continuationのtoken log probabilityを合計する。

- primary: preferred勝数 / 10、preferred合計log-prob − dispreferred合計log-prob。
- secondary: token平均NLLとperplexity（長さ差の診断用）。
- Stage 2 / Stage 3で同一pairとtokenizerを使う。

記事に掲載された過去10 pair全文は回収できないため、記事のStage 2 `+11.86`、Stage 3 `+12.26`と新しい数値は直接比較しない。計算方法の再現と、今回以後の共通baseline固定を目的とする。
