# M0 Stage 2 / Stage 3 baseline再評価プロトコル

更新日: 2026-08-20

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

## 2026-08-18〜20に判明した3点

実行記録は`../reports/m0-baseline-final.json`。prompt準備とcheckpoint checksumは初回から通過したが、生成と口調評価はそれぞれ別の理由で止まり、3件とも解消した。

### 1. 生成: 公開checkpointは`dep_q=8`で`generate.py`が動かなかった

公開Stage 2/3の`moshi_lm_kwargs.json`は`n_q=16`、`dep_q=8`、`depformer_context=8`である。`tools/clean_moshi.py`を`--remove_modules_for_user_stream`付きで通した推論用形式で、user streamは入力としては扱うが生成しない。一方`models/moshi_for_generation.py`の`step`は`1 + dep_q = 9` tokenを作って`num_codebooks = 1 + n_q = 17`と一致するかassertするため停止した。

**解決（ユーザー承認）: user streamを無音で教師強制する。** `MoshiForConditionalGeneration`に`num_user_codebooks`を追加し、`generate.py`はdelay適用済み`batch.input_ids`から該当streamを1 frameずつ供給する。prompt音声は`prompt_length + generation_length + 2` frameぶんまで無音でpaddingする。

したがって本baselineは「**user streamは無音を教師強制し、Moshi側のstreamだけを生成する**」条件と定義する。B channelは元から無音指定なのでprotocolの意図と矛盾しない。生成音声20件すべてでB channelのpeakが`402`一定であることが、全frameで教師強制が効いた証拠になる。

### 2. checkpoint読み込み: 公開weightはoriginal Moshi名だった

公開weightは`clean_moshi.py`がoriginal Moshi名（`gating.linear_in.weight`）で保存しており、`MoshiForFinetuning.__init__`のDeepSpeed Zero-3向け改名（`gating.linear_in_weight`）と166 parameterで一致しなかった。`tools/moshi_state_dict.py`に双方向の名前対応を実装して解消した。`persona_perplexity`と`generate.py`の両方をブロックしていた。

### 3. 口調perplexity: 原因は勝敗の判定式だった

3回の試行で絶対NLLは一様分布（`log(32000) = 10.373`）を一度も下回らなかった。

| 試行 | 条件 | Stage 2 | Stage 3 |
| --- | --- | ---: | ---: |
| 1 | 全codebookを`zero_token_id` | `12.878` | 未実行 |
| 2 | 実Mimi token条件 | `13.004` | `15.204` |
| 3 | 時刻整列を模したtext layout | `14.201` | `13.207` |

音声条件も整列も絶対値を動かすだけで答えを変えなかった。**絶対NLLが高いのは構造的な性質**である。Moshiのtext streamは自由な言語モデルではなく、与えられた音声の書き起こしに対応する。条件音声は採点テキストと別の発話なので、モデルは音声側の書き起こしを予測し、こちらのtokenには低い確率しか与えない。

しかし対比較では、両候補が**同じ文脈・同じ音声・同じlayout**を共有する。違いはcompletion tokenだけなので、比較は成立する。実際に問題だったのは勝敗の判定式だった。

- 旧: 合計log-probで比較 → 長い候補が不利。preferredが長い候補のpairが10件中6件あり、**3/10**
- 新: 長さ正規化した平均NLLで比較 → **Stage 2 = 7/10、Stage 3 = 7/10**

これは過去記事の「お嬢様選好 7/10」（軽量版・フル版とも）と一致する。記事の`+11.86 / +12.26`は別baselineに対する別量のため比較しない。

### 撤回したGate

当初、`preferred_mean_nll`が`log(text_card)`を下回らないrunを失敗させる一様分布ゲートを入れた。この前提は対比較に対して誤っており、**機能している指標を3回却下した**。`assert_scores_discriminate`へ置き換え、非有限値と、全pairで両候補が同点になる場合（＝completion tokenが採点位置に届いていない、実際に壊れている形）だけを失敗させる。

絶対値の妥当性ではなく、候補が結果を動かせているかを見るのが、対比較に対する正しい検査である。

## 口調perplexity（確定方式）

`../eval/persona-baseline-10.jsonl`の10 pairを`reconstructed-v1`として固定する。rinna SentencePiece（`rinna/japanese-gpt2-medium` `spiece.model`、revision `f464b767…`）を使う。これはREADMEの99〜105行がJ-Moshiの採用tokenizerとして明記しているもので、学習時の`encode_as_pieces_wo_byte_fallback` + `piece_to_id`と本評価の`encode(out_type=int)`が10 pair全30文字列で同一IDを返すことを確認済み。

採点条件:

- 音声条件はheld-out promptの実Mimi token（`VOICEACTRESS100_032`、A 8 + B 8 = 16 codebook）。各codebookに`delays`を適用し、delay前のframeは`initial_token_id`で埋める
- text streamは`text_padding_token_id`で埋め、`start_frame = 12`から連続frameへcontextとcompletionを置き、直前1 frameに`end_of_text_padding_id = 0`を入れる。`tools/tokenize_text.py`のlayoutに合わせる
- 先頭に`delay_and_pad_streams`と同じinitial token frameを1つ付ける
- Stage 2 / Stage 3で同一pair・同一tokenizer・同一音声条件を使う

指標:

- primary: **長さ正規化した平均NLLによるpreferred勝数 / 10**。合計log-probは長い候補を不利にするため使わない
- secondary: 平均NLL margin、perplexity、合計log-prob（診断用に併記）

検査は`assert_scores_discriminate`が行う。非有限値と、全pairで両候補が同点になる場合だけ失敗させる。絶対NLLの水準は判定に使わない。条件音声が採点テキストと別の発話である以上高くなるのが構造的であり、両候補が同じ条件を共有する対比較には影響しないためである。

記事に掲載された過去10 pair全文は回収できないため、記事の`+11.86 / +12.26`とは直接比較しない。ただし記事の選好7/10（軽量版・フル版とも）は本方式で再現できている。
