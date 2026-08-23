# J-Moshi-ext を「つくよみちゃん系の声 × お嬢様口調」に適応する実験計画

更新日: 2026-08-21

ステータス: M0、M1、M2は完了。M3（Voice control）は**評価を完了したが不合格**。
V0/V1のいずれも採用基準を満たさず、M4へ渡すcheckpointがないためM4はBlocked。
判定と証拠は[マイルストーン文書](./j-moshi-tsukuyomi-ojousama-milestones.md)、経緯の通読は[M3実施報告](./j-moshi-tsukuyomi-ojousama-m3-report.md)、
実行順序・ゲート・費用は[M3実行計画](./j-moshi-tsukuyomi-ojousama-m3-plan.md)を正とする。

**GPU予算US$100を超過した（累計US$102.70）。** 追加のGPU作業は新しい上限の承認を要する。

作業ブランチ: `experiment-j-moshi-character-voice-overfit`

この文書を技術方針とrun条件の実行基準とする。進捗、成果物、マイルストーンごとの目的・ゴール・完了条件は[マイルストーン文書](./j-moshi-tsukuyomi-ojousama-milestones.md)を正とする。調査前に作成した汎用のvoice overfit案は、過去実験の記録と矛盾する部分があったため本書へ統合し、個別文書としては残さない。

## マイルストーン管理

本実験はM0〜M6で管理する。各マイルストーンには、目的、ゴール、入力、作業、成果物、完了条件、次へ進む条件、完了記録を定義している。

| ID | 概要 | 現在の状態 |
| --- | --- | --- |
| M0 | 過去実験・Vast.ai基盤 | 完了 |
| M1 | 権利・データ確定 | 完了 |
| M2 | Tsukuyomi TTS | 完了 |
| M3 | Voice control | 完了（不合格） |
| M4 | Voice overfit | Blocked |
| M5 | お嬢様口調 | 未着手 |
| M6 | 最終検証 | 未着手 |

状態と完了証拠は[マイルストーン文書](./j-moshi-tsukuyomi-ojousama-milestones.md)だけで更新し、この概要表は構成変更時に同期する。

## 確定した判断

| 論点 | 決定 |
| --- | --- |
| Base | 過去比較を優先し、まず `nu-dialogue/j-moshi-ext` を使用する |
| Voice 学習形式 | A=対象話者、B=中立話者の 2 話者 stereo。1 話者朗読の直接学習を主経路にしない |
| Voice 更新範囲 | 全パラメータ。depformer-only は過去の mode collapse により除外 |
| Voice control LR | tempformer / depformer ともに `3e-5` |
| Voice overfit | control 合格後、tempformer `3e-5`を固定して depformerを`6e-5`、次に`1e-4`へ上げる |
| Style 更新範囲 | paired stereo を入力し、tempformer-only |
| Style control LR | `3e-5` |
| お嬢様 script | 過去100対話を回収。なければ100対話だけ再生成し、最初から1,000対話にはしない |
| 合成音声 | 学習形式上必要。COEIROINK出力は使わず、利用可能な原音から作る独自TTSを使用 |
| 中立話者B | VoiceDesignのcaptionから1本だけ生成し凍結、以降は`--ref-wav`で再利用。発話ごと生成は実測で話者が定まらない |
| Voice GPU構成 | 2× A100 80GB、ディスク900 GB。1枚では常駐133.94 GBに対し48.04 GB不足する |
| 評価 | lossだけで採用しない。声質、明瞭度、口調、full-duplex対話を別評価する |

## 結論

過去実験から、今回の基本構成は次の順序に修正する。

1. つくよみちゃん原音から対象話者 TTS を作り、単体で声質・明瞭度を評価する。
2. その TTS と中立話者 TTS で、J-Moshi の学習分布に合う 2 話者 stereo 対話を作る。
3. J-Moshi-ext を **全パラメータ**で voice fine-tuning する。
4. voice checkpoint を基準に、お嬢様対話で **tempformer-only** fine-tuning する。
5. 声質・口調・明瞭度・turn-taking を別々に評価する。

過去に depformer-only を朗読データへ適応した実験では、audio loss は低下したが、ユーザーを無視する独話ループ、発話反復、音声アーティファクトが発生した。この失敗が再現条件まで記録されているため、今回の主経路から depformer-only を除外する。

お嬢様口調については、新たに 1,000 対話を作る必要はない。過去に **100 dialog / 800 turn / 約 40 分**で効果が確認されている。まず過去の 100 対話 script を回収し、見つからない場合だけ同規模の 100 対話を再生成する。ただし、文字列だけでは現在の Moshi 学習形式に入らないため、同じ script をつくよみちゃん系 TTS で音声化した paired dataset は必要である。

## 過去実験から回収できた情報

情報源:

- [J-Moshi-ext に「あみたろ」の声を追加学習する](https://ayousanz.hatenadiary.jp/entry/2026/07/07/193000)
- [J-Moshi-ext に「お嬢様口調」を追加学習する](https://ayousanz.hatenadiary.jp/entry/2026/07/07/230633)

### Voice fine-tuning

| 項目 | 過去の値・結果 |
| --- | --- |
| Base | `nu-dialogue/j-moshi-ext`、7.5B、`dep_q=16` |
| 話者原音 | あみたろコーパス ITA 2.2 + MANA、実 189.9 分（3.16 時間） |
| TTS | Irodori-TTS 500M v3 を全パラメータ fine-tuning |
| 合成対話 | 100 dialog、612 turn、31.7 分、24 kHz stereo |
| 話題 | 天気、食、仕事、家族、旅行、趣味、健康、買い物、技術、雑談 |
| 話者 | A = あみたろ TTS、B = Irodori base の別話者 |
| Moshi 更新範囲 | 全パラメータ |
| 学習率 | `3e-5`、AdamW |
| batch | A100 80 GB ×2、per-device 1、gradient accumulation 4、effective batch 8 |
| 学習長 | 5 epochs、65 steps、28 分 |
| loss | `12.53 → 1.24` |
| 結果 | 声質は一部転移、full-duplex 対話は保持、発音の明瞭度は低下 |

この結果から、今回の voice stage では `3e-5` を「過去に動いた control」とする。論文由来の小さい fine-tuning 学習率だけを基準にせず、同一データ形式・同一モデルで実績のある値を最初に再現する。

### 失敗した depformer-only

過去には実あみたろ corpus 2.58 時間と合成 320 対話を使った depformer-only fine-tuning も実行されている。

- audio loss は `6.88 → 1.02` まで低下した。
- ユーザー入力を無視する独話ループが発生した。
- 「ありがとうございました」等の反復と mode collapse が発生した。
- 声は対象話者ではなく、汎用女性声と強いアーティファクトになった。

loss 低下だけでは成功判定にならないこと、1 話者朗読をそのまま Moshi に入れると 2 話者対話の学習分布と衝突することが確認されている。したがって、今回「音声側を強める」とは、depformer だけを更新することではなく、**2 話者 paired dataset の全パラメータ学習を維持したまま、depformer 学習率を相対的に上げる比較**を意味する。

### お嬢様口調 fine-tuning

| 項目 | 過去の値・結果 |
| --- | --- |
| Base | `ayousanz/phase1b-jmoshi-ft-2026-07-06`（あみたろ voice checkpoint） |
| script | 10 topics × 10 dialog、100 dialog、平均 8 turn、計 800 turn |
| text 生成 | Claude Sonnet。指定語尾の出現率 90% 以上を正規表現で検査 |
| 音声 | A = お嬢様口調のあみたろ TTS、B = 中立の執事役、24 kHz stereo、約 40 分 |
| 軽量版 | tempformer-only、`3e-5`、5 epochs、約 30 分 |
| フル版 | 全 7.5B、その他は軽量版と同じ |
| 軽量版結果 | text loss `0.07`、お嬢様選好 7/10、基準から log-prob +83% |
| フル版結果 | text loss `0.30`、お嬢様選好 7/10、基準から log-prob +89% |
| 判断 | tempformer-only が全パラメータ版とほぼ同等で、声を保護できた |

この実績により、口調 stage の第一候補は loss mask を新設した文字列-only 学習ではなく、過去と同じ paired stereo dataset を使う `--parameters_to_finetune tempformer` である。

## Git・ローカル成果物の回収結果

2026-08-18 に以下を確認した。

| 確認先 | 結果 |
| --- | --- |
| ローカル branch | `main`、今回の実験 branch、無関係な調査 branch のみ |
| `origin` branch | `main` のみ |
| tag / stash | なし |
| GitHub PR refs / PR 一覧 | なし |
| GitHub release / fork | なし |
| reflog | 2026-08-13 の clone 以降だけ。2026-07-07 の実験履歴なし |
| 全 commit tree のキーワード検索 | 過去の config、manifest、合成 script なし |
| Desktop / Downloads | Irodori-TTS とあみたろ原音は存在。過去の合成対話 directory はなし |
| Hugging Face cache | Stage 3 の ref SHA のみ。ローカル重み snapshot はなし |

現在の `main` にある次の 2 commit は、過去記事に書かれたトラブル対応と一致する。

- `d914a79`: partial fine-tuning 時の empty parameter group を除外
- `14cd770`: tokenization に失敗した dialog を skip

過去の成果 checkpoint は Git branch ではなく Hugging Face にある。

- Voice checkpoint: [`ayousanz/phase1b-jmoshi-ft-2026-07-06`](https://huggingface.co/ayousanz/phase1b-jmoshi-ft-2026-07-06) — 現在は認証が必要
- Tempformer-only お嬢様: [`ayousanz/moshi-persona-stage2-ojousama-2026-07-06`](https://huggingface.co/ayousanz/moshi-persona-stage2-ojousama-2026-07-06) — 公開
- Full fine-tuning お嬢様: [`ayousanz/moshi-persona-stage3-ojousama-2026-07-06`](https://huggingface.co/ayousanz/moshi-persona-stage3-ojousama-2026-07-06) — 公開

したがって、回収対象は別 branch ではなく、過去に学習したサーバ・クラウドストレージ・W&B artifact に残っている可能性がある次のファイルである。

- `synthetic-dialogs-2026-07-04/`
- `ojousama_mild_100/`
- `dialogs_source.jsonl`
- `manifest.jsonl`
- `configs/j-moshi-ext-amitaro.yaml`
- `configs/persona-tempformer.yaml`
- `configs/persona-full.yaml`
- `persona_perplexity.py`

## 今回のデータ設計

### 1. つくよみちゃん原音

| 候補 | 内容 | 判断 |
| --- | --- | --- |
| [つくよみちゃんコーパス](https://tyc.rei-yumesaki.net/material/corpus/) | 公式原音 100 文 | 最初の TTS / 実音 pilot に使う |
| [夢前黎の音声データの寄せ集め](https://tyc.rei-yumesaki.net/material/corpus/yoseatsume/) | 申請制、約 1,500 台詞 | 取得できれば優先的に追加する |
| COEIROINK 生成音声 | 任意文を生成可能 | [規約](https://coeiroink.com/terms)上、機械学習データには使わない |

公式 100 文は、過去のあみたろ原音 3.16 時間より大幅に少ない。J-Moshi の学習率より先に、Irodori-TTS がこの量で声と明瞭度を再現できるかが最大のゲートになる。

### 2. Voice dataset の二案

M3 着手時の実測で、当初案から三点を変更した。確定した実行条件は
[M3実行計画](./j-moshi-tsukuyomi-ojousama-m3-plan.md)にある。

#### V-real: 実音を中心にした 80 対話

- A の発話はつくよみちゃん公式原音をそのまま使う。
- 各原音に対して自然につながる B の前後発話を作り、中立 TTS で音声化する。
- A=左、B=右の stereo 対話にする。
- A の声質は TTS 品質に制約されないが、A の内容は元の文に限定される。

**100 文ではなく train 分割の 80 文に限る。** コーパスは train 80 / dev 10 / test 10 に
分割済みで、固定評価セット `eval/voice-seen-heldout-20.jsonl` の held-out 側は test 由来である。
100 文すべてを学習に入れると M3 完了条件4「held-out 音声の話者らしさ」が測れなくなる。
結果として A の音声は 8.67 分となり、過去のあみたろ 189.9 分の 4.6% にとどまる。

これは過去記事が代替案として挙げた「Speaker A のみ実 corpus、Speaker B は別 TTS」を具体化したものになる。

#### V-tts: TTS で作る 80 対話

- A は M2 で採用した Irodori-TTS speaker inversion（T1）で生成する。全パラメータ
  fine-tuning は M2 で不採用となった。
- 一般対話 script は新規作成する（過去の 100 script は回収不能。`m0/artifact-recovery.md`）。
- A をつくよみちゃん TTS、B を中立 TTS で生成する。

**当初の 600〜800 turn・30〜40 分ではなく、V-real と同じ 80 対話 240 turn に揃える。**
規模を揃えないと V0 と V1 の差が「実音か TTS か」ではなく「データ量の差」になり、比較が成立しない。
V-real と V-tts は script・分割・話者B音声・base・依存をすべて共有し、**違うのは
チャンネルAのバイト列だけ**にする。

Irodori-TTS の有声区間について主観評価、誤読率、発音の明瞭度、話者類似度を確認し、不合格なら V-tts を Moshi に投入しない。V-real と V-tts は最初から混ぜず、個別の効果を比較する。

#### 中立話者 B

Irodori-TTS-600M-v3-VoiceDesign の caption 条件（`--no-ref --caption`）で **1 本だけ生成し、
それを凍結して以降のすべての B 発話の `--ref-wav` に使う**。

発話ごとに caption から生成する方式は実測で否定された。pairwise ECAPA が 0.4525（seed 既定）
/ 0.5176（seed 固定）となり、実人間一人の下限 0.5646 を下回る。これは「似ているが別々の声の
集合」であって一人の話者ではなく、そのまま使えばチャンネルB が話者ではなく群衆になる。
凍結すると 0.7373 / 最小 0.7040 で人間の帯を上回り、話者 A との分離は 0.157 となる。
測定は [`m3-speaker-b-probe.json`](../../experiments/tsukuyomi_ojousama/reports/m3-speaker-b-probe.json)。

参照音声を第三者から持ち込まないため、追加のライセンス義務は発生しない
（VoiceDesign 自体の条項は M3 ステップ4で確認する）。

### 3. お嬢様 dataset

優先順位は次の通りとする。

1. 過去の `ojousama_mild_100` script を回収する。
2. 回収できない場合、ブログに記録された 10 topics × 10 dialog を同じ schema で再生成する。
3. `OjousamaTalkScriptDataset` 200 組は口調の参考、品質監査、held-out 評価に使う。
4. 100 対話で未学習話題への口調転移が不足した場合のみ追加生成する。

過去 script を回収できても、A の音声はあみたろではなくつくよみちゃん系 TTS で再生成する。過去のあみたろ音声をそのまま Tsukuyomi voice checkpoint の style stage に入れる方法は、tempformer の音声条件を別話者へ寄せる可能性があるため control にはしない。

口調規則は、過去条件を再現する strict 版と、自然さを優先する mild 版を分ける。

| Variant | 語尾条件 | 目的 |
| --- | --- | --- |
| S-strict | 指定語尾を含む A 発話 90% 以上 | 過去結果の再現 |
| S-mild | 60〜75%、同一語尾の連続禁止 | 語尾過多を抑えた自然な会話 |

## 学習計画

### Stage 0: 過去 checkpoint の baseline

- 公開 Stage 2 / Stage 3 を同じ live prompt で動かす。
- 可能なら認証付き Voice checkpoint も取得する。
- お嬢様選好 10 pair の評価を再現する。
- 過去の問題である応答遅延、短い応答、話題の浅さ、発音劣化を音声で保存する。

これにより、今回のモデルが「過去より良いか」をブログの記憶ではなく同じ評価器で比較できる。

### Stage 1: Tsukuyomi TTS gate

- 公式 100 文で Irodori-TTS を対象話者へ適応させる。
- train 文だけでなく、未学習のお嬢様語彙を 30 文生成する（`eval/tts-unseen-30.jsonl`）。
- 誤読、音の欠落、話者類似性、抑揚を評価する。
- 追加 1,500 台詞を取得できた場合は同条件で再学習して比較する。

TTS が不明瞭なまま Moshi 学習へ進むと、過去と同じくデータ品質が J-Moshi の上限になる。

#### 適応方式を全パラメータ学習から変更した理由（2026-08-20）

当初は過去のあみたろ実験に合わせて全パラメータ fine-tuning を予定していた。しかし実測したデータ量が前提と大きく異なる。

| | あみたろ（過去） | つくよみちゃん（今回） |
| --- | ---: | ---: |
| 総量 | 189.9 分 | 10.97 分 |
| train | — | 8.67 分 |
| 比率 | 100% | 5.8% |

8.67 分で 500M モデルの全パラメータを更新すれば過学習は避けられず、これは完了条件「train 文の暗記だけでなく未学習のお嬢様語彙を発音できる」が排除しようとしている失敗そのものである。Irodori-TTS はこの規模に適した方式を用意しているため、比較の順序を次のとおりとする。

| 段階 | 方式 | 位置付け |
| --- | --- | --- |
| T0 | zero-shot voice cloning（学習なし） | base TTS control。完了条件「話者らしさが base TTS より改善」の比較対象 |
| T1 | Speaker Inversion（base 凍結、話者埋め込みのみ学習） | 第一候補。単一話者・少量データ向けに用意された方式 |
| T2 | LoRA | T1 が不足した場合 |
| T3 | 全パラメータ | T1/T2 が不足した場合のみ。過学習の確認用 |

zero-shot が control になるため、T0 は方式比較の基準として必ず取る。M2 の目的は「データ品質の上限を確認する」ことなので、量に見合わない方式で失敗させるのではなく、量に見合った方式で到達点を測る。

推論は Apple Silicon の MPS で動作する（1 文あたり約 17 秒）ため、生成だけなら GPU レンタルは不要である。

### Stage 2: Voice fine-tuning

全 run で全パラメータを更新し、2 話者 stereo dataset を使う。`--model_user_stream` を有効にし、両話者の対話分布を維持する。

| Run | dataset | tempformer LR | depformer LR | 位置付け |
| --- | --- | ---: | ---: | --- |
| V0 | V-real | `3e-5` | `3e-5` | 実音中心、過去 LR control |
| V1 | V-tts | `3e-5` | `3e-5` | 過去 pipeline の再現。V-real と規模を揃える |
| V2 | V0/V1 の良い方 | `3e-5` | `6e-5` | 音声側を 2 倍にする overfit |
| V3 | V0/V1 の良い方 | `3e-5` | `1e-4` | V2 が安定・改善した場合だけ |

最初から V2/V3 を実行しない。V0/V1 で mode collapse がなく、過去 control と同程度の対話能力を確認してから学習率を上げる。5 epochs を上限とし、最終 checkpoint だけでなく各 epoch の checkpoint を比較する。

現行 `finetune.py` では、少なくとも次を run ごとに明示する。

```text
--moshi_speakers A
--model_user_stream
--parameters_to_finetune all
--tempformer_learning_rate <表の値>
--depformer_learning_rate <表の値>
--num_train_epochs 5
--text_padding_loss_weight 0.5
--semantic_loss_weight 100.0
--acoustic_loss_weight 1.0
```

過去 config から scheduler を回収できるまでは、control は現行実装の既定値である `--num_warmup_steps 0` を明示する。過去値が判明した場合は、その値を再現 run にのみ使用し、別 run として記録する。

設定選択用データは対話単位で 72/8 に分割する（対話単位の test は作らない。held-out 評価は
コーパス test 分割の音声で行う）。`S` は起動ログを待たずに**オフラインで確定して起動時に固定する**。
起動ログは `S` の確認にのみ使い、想定と違えば即座に kill する。`--save_steps S`、`--eval_steps S`
で各 epoch を保存・評価する。最良設定を固定した後だけ、全対話を使う final-overfit run を
別 output directory で実行する。

#### 実行環境の確定事項

M3 着手時にコードから確定した。いずれも見積もりではなく計算値である。

| 項目 | 値 | 影響 |
| --- | --- | --- |
| dep_q=16 モデル | 8,371,408,896 パラメータ | — |
| 学習時の常駐 | 16 bytes/param = 133.94 GB | **1× A100 80GB（85.90 GB）では 48.04 GB 不足。2 枚必須** |
| ホスト RAM | 80 GB 以上 | `finetune.py` が rank ごとに fp32 で CPU ロードする。不足すると起動時に OOM kill され、GPU をいくら積んでも解決しない |
| ZeRO-3 checkpoint | 12 bytes/param = 100.46 GB / 本 | fp16 の重みコピーはディスクに書かれない |
| checkpoint ローテーション | **存在しない** | `--save_total_limit` が無く、accelerate の自動命名も無効。5 epoch で 502 GB が積み上がる |
| ディスク | 900 GB | 300 GB では V0 の 2 本目の書き込み途中で死ぬ |

### Stage 3: お嬢様口調 fine-tuning

Stage 2 の最良 voice checkpoint を base にする。

| Run | dataset | 更新範囲 | LR | 目的 |
| --- | --- | --- | ---: | --- |
| S0 | S-strict 100 dialog | tempformer-only | `3e-5` | 過去結果の再現 |
| S1 | S-mild 100 dialog | tempformer-only | `3e-5` | 自然さとの比較 |
| S2 | S0/S1 の良い方 | all | `3e-5` | tempformer-only で不足した場合のみ |

過去実験では tempformer-only と all の差が小さかったため、S2 は原則不要である。声を保護し、コストを下げられる S0/S1 を優先する。

S0/S1 では次を明示し、Voice stage の最良 checkpoint 以外を変更しない。

```text
--moshi_speakers A
--model_user_stream
--parameters_to_finetune tempformer
--tempformer_learning_rate 3e-5
--num_train_epochs 5
```

tempformer-only は text-only dataset を意味しない。音声 loss からも tempformer へ勾配が流れる実装であるため、A/B の音声、発話時刻、テキストが一致した paired stereo を必須とする。

## Gate と中止条件

| Gate | 合格条件 | 不合格時 |
| --- | --- | --- |
| TTS | 未学習30文で欠落・クリップなし、明瞭に読める文が27/30以上、主観的な話者らしさがbase TTSより改善 | Moshiへ投入せず、追加音声またはTTS設定を改善 |
| Voice control | 固定30会話で独話loop・反復collapseなし、明瞭度がJ-Moshi-extから大きく低下しない | LRを上げず、dataset/TTSを修正 |
| Voice overfit | held-out声質がcontrolより改善し、明瞭度・turn-takingがcontrol同等 | 最良の低LRまたは中間checkpointへ戻す |
| Style | held-out 50 pairでcontrolよりお嬢様選好が改善し、同一語尾の連続と応答品質が許容範囲 | 追加量ではなくstrict/mild比率とscript品質を修正 |
| 最終 | 声質、明瞭度、口調、full-duplexの全条件を同時に満たす | 単一loss最小のcheckpointを採用しない |

## 評価

### 声質

- target 原音と生成音の有声区間だけを抽出し、RMS 正規化後に speaker embedding 類似度を計算する。
- 自動類似度だけで採用せず、日本語話者によるブラインド A/B を行う。
- 音高、響き、話速、抑揚、明瞭度を別項目で採点する。

過去実験では、SECS が高い棒読み TTS より、SECS が低い full fine-tuning TTS の方が主観的に本人らしい事例があった。SECS 単独評価は禁止する。

### お嬢様口調

- 過去の 10 pair perplexity を再現し、評価を 50 pair 以上へ拡張する。
- `わたくし`、`ですわ`、`かしら`等の選好だけでなく、同じ語尾の連続率も測る。
- 未学習の相談、説明、雑談で口調が維持されるか確認する。
- strict と mild を run 名を伏せて比較する。

### Full-duplex 対話

- ユーザー発話への追随時間。
- 応答開始までの遅延。
- 独話ループ、同文反復、無視、長い無音。
- 応答長、話題の深さ、質問への関連性。
- 割り込みと相づち。

train/audio loss が低くても、live 対話のいずれかで collapse した checkpoint は不採用とする。

## 実行ロードマップ

以下は技術的な実行順序の要約である。プロジェクト上の完了判定には[マイルストーン文書](./j-moshi-tsukuyomi-ojousama-milestones.md)のチェックリストを使う。

1. **外部artifact回収**: 過去サーバ・クラウド・W&B artifact から config、script、manifest、`persona_perplexity.py`を探す。ローカルGit探索は繰り返さない。
2. **過去baseline固定**: 公開 Stage 2 / Stage 3 checkpoint の生成音声と perplexity を同一promptで保存する。
3. **データ台帳**: つくよみちゃん原音、派生TTS音声、会話script、checkpointの出典・版・利用条件・クレジットをmanifest化する。
4. **TTS gate**: 公式100文でTsukuyomi TTSを作り、未学習30文を評価する。可能なら約1,500台詞を取得した版も比較する。
5. **Voice dataset**: V-realとV-ttsを別々に作り、80/10/10分割・tokenize・parquet化を検証する。
6. **Voice control**: V0/V1を全パラメータ`3e-5`で実行し、各epochとlive対話を比較する。
7. **Voice overfit**: control合格後だけV2、さらにV2合格後だけV3を実行する。
8. **Style dataset**: 過去100対話を回収または再生成し、つくよみちゃん系TTSでS-strict/S-mildを音声化する。
9. **Style fine-tuning**: 最良voice checkpointからtempformer-only S0/S1を実行する。
10. **Final overfit**: 採用設定を100対話全体で再学習し、固定評価に合格したcheckpointだけを公開候補にする。

各段階は前の Gate に合格してから進む。特に、TTSが不明瞭な状態でJ-Moshiの学習率探索へ進まない。

## 参照

- [過去の voice fine-tuning 記録](https://ayousanz.hatenadiary.jp/entry/2026/07/07/193000)
- [過去のお嬢様口調 fine-tuning 記録](https://ayousanz.hatenadiary.jp/entry/2026/07/07/230633)
- [つくよみちゃんコーパス](https://tyc.rei-yumesaki.net/material/corpus/)
- [夢前黎の音声データの寄せ集め](https://tyc.rei-yumesaki.net/material/corpus/yoseatsume/)
- [COEIROINK 利用規約](https://coeiroink.com/terms)
- [OjousamaTalkScriptDataset](https://github.com/matsuvr/OjousamaTalkScriptDataset)
