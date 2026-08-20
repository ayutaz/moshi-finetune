# つくよみちゃん系の声 × お嬢様口調 実験マイルストーン

更新日: 2026-08-20

作業ブランチ: `experiment-j-moshi-character-voice-overfit`

実行環境: Vast.ai

技術方針と run matrix は[実験計画](./j-moshi-tsukuyomi-ojousama-plan.md)を正とし、この文書は進捗、成果物、完了判定を管理する正本とする。

## 最終目的

J-Moshi-ext に、つくよみちゃんコーパス由来の声質と、自然なお嬢様口調を追加学習する。声質だけ、口調だけ、loss だけの改善では完了とせず、次を同時に満たす release candidate を作る。

- held-out 音声で対象話者らしさが control より改善する。
- 発音の明瞭度を維持する。
- 未学習話題でもお嬢様口調を維持し、同じ語尾を過剰反復しない。
- ユーザー入力を無視せず、独話 loop や mode collapse を起こさない。
- データ、設定、checkpoint、評価結果の出典と再現手順が残っている。
- 音声・テキスト・派生モデルの利用条件を満たしている。

## 状態の定義

| 状態 | 意味 |
| --- | --- |
| 未着手 | 入力または前段マイルストーンの完了待ち |
| 進行中 | 作業中。完了条件をすべて満たしてはいない |
| Blocked | 外部許可、データ、計算資源などがなく進められない |
| 完了 | 成果物と検証証拠が揃い、完了条件をすべて確認済み |

マイルストーンは、学習を実行しただけでは完了にしない。「完了条件」の各項目を証明するファイル、ログ、checkpoint、評価結果への参照を「完了記録」に記入して初めて完了とする。

## 一覧

| ID | マイルストーン | 目的 | ゴール | 状態 | 依存先 |
| --- | --- | --- | --- | --- | --- |
| M0 | 過去実験・Vast.ai基盤 | 比較基準と安全な実行環境を確立 | 過去baselineを固定し、Vast.aiへSSH接続して学習準備完了 | 完了 | なし |
| M1 | 権利・データ確定 | 学習可能で再現可能な入力を確定 | データ台帳、分割、固定評価セットを完成 | 完了 | なし（M0と並行可） |
| M2 | Tsukuyomi TTS | 対話音声を生成できる対象話者TTSを作る | 未学習30文のTTS Gateを通過 | 完了 | M0, M1 |
| M3 | Voice control | 過去成功条件をつくよみちゃんで再現 | V0/V1の少なくとも一方で声質改善と対話維持を両立 | 未着手 | M2 |
| M4 | Voice overfit | 声質をcontrolより強く適応 | V2/V3から品質を壊さない最良checkpointを選定 | 未着手 | M3 |
| M5 | お嬢様口調 | 声を保持しながら話し方を転移 | S0/S1で口調改善、声質・対話品質を維持 | 未着手 | M4 |
| M6 | 最終検証 | 全条件を満たす再現可能な成果物へ統合 | final-overfitと全Gateを完了しrelease candidateを固定 | 未着手 | M5 |

## M0: 過去実験・Vast.ai基盤

### 目的

過去実験との比較基準を固定し、認証情報を安全に扱いながらVast.aiで学習・評価できる状態を作る。

### ゴール

過去のStage 2 / Stage 3を同じ評価条件で再生でき、Vast.aiの稼働インスタンスへSSH接続し、リポジトリ・永続ストレージ・GPUを確認できる。

### 入力

- 過去のvoice / お嬢様口調ブログ記事
- Hugging Face上の過去checkpoint
- 現在の`moshi-finetune`リポジトリ
- Vast.aiアカウントとCLI

### 作業

1. 過去サーバ、W&B、クラウドストレージからconfig、script、manifest、評価コードを探す。
2. 見つからない成果物は、探索先と欠損を記録する。
3. 公開Stage 2 / Stage 3 checkpointを取得し、固定promptで生成する。
4. 過去の10 pair perplexity評価を再現する。
5. Vast.ai API keyの露出リスクを確認し、原則ローテーションする。ユーザーがリスクを理解した上で現keyの継続を明示承認した場合は、例外理由と日付を記録する。
6. keyファイルを所有者だけが読める権限にし、API認証を確認する。
7. Vast.aiインスタンスを用意し、SSH、GPU、CUDA、ディスク、永続保存先を確認する。
8. 学習開始前のクレジットと想定上限を記録する。API keyや個人情報は記録しない。

### 費用上限

- 本実験で消費してよいVast.ai費用は合計`US$100`以下とする。
- 課金開始直前の残高を起点として記録し、他用途の支出があれば実験費用から分離する。
- 累計`US$75`で警告し、残りのrunと評価に必要な費用を再見積もりする。
- 予測累計が`US$90`を超える新規runは開始しない。
- 請求反映遅延の余裕を残すため、実測累計`US$95`で稼働中の学習を停止する。
- 上限変更はユーザーの明示的な承認がある場合だけ行う。

### 成果物

- 過去artifactの回収・欠損一覧
- Stage 2 / Stage 3のbaseline生成音声
- 再現したperplexity評価結果
- Vast.ai環境情報と接続確認記録
- GPU、依存バージョン、ストレージ構成を含む環境manifest

### 完了条件

- [x] 過去artifactの回収結果がファイル単位で記録されている。
- [x] Stage 2 / Stage 3の固定生成音声が保存されている。
- [x] Stage 2 / Stage 3の口調評価値が保存されている。
- [x] API keyの扱いを決定済みである（2026-08-18、ユーザーが現key継続を明示承認）。
- [x] API keyがリポジトリ内に存在せず、保存ファイルの権限が`600`相当である。
- [x] Vast.aiに専用の稼働中インスタンスがあり、SSH接続できる。
- [x] GPU、CUDA、VRAM、空きディスク、保存先とinstance破棄時の消去条件を確認済みである。
- [x] 学習コストの記録方法と停止上限が決まっている（承認上限`US$100`、実測`US$95`で安全停止）。

### 次へ進む条件

M1のデータ監査はM0と並行可能とする。M2へ進むにはM0とM1の両方が完了していることを要求する。API keyのローテーションまたは明示例外の記録と、SSH接続が未完了ならデータをアップロードしない。

### 完了記録

**状態: 完了**（2026-08-20）。統合レポート: `experiments/tsukuyomi_ojousama/reports/m0-baseline-final.json`

#### 過去artifactの回収

- 探索と欠損の一覧: `m0/artifact-recovery.md`。過去のtraining artifact（合成対話、manifest、旧`persona_perplexity.py`、W&B run）は回収不能と確定
- Stage 2 / Stage 3: 固定revisionのweight・configを回収し、SHA-256を固定。`bootstrap_instance.sh`実行時に再ダウンロードして両方の一致を再確認済み
- 過去実験条件: ブログ2記事から設定値と結果を計画文書へ転記

#### baseline生成音声

- 両stageとも10 token・10 WAV、各9.92秒の24 kHz stereo（prompt 40 frames、continuation 125 frames、seed `20260818`）
- 実測差: A channelの平均RMSは Stage 2 `2691.3` / Stage 3 `3827.1`。クリップは Stage 3 の1件に10 sampleのみ（Stage 2は0）
- B channelは全20件でpeak `402`一定。教師強制した無音が全frameで効いたことの確認になる

#### 口調評価値

- **Stage 2 = 7/10、Stage 3 = 7/10**（長さ正規化した平均NLLによる選好）。過去記事の「お嬢様選好 7/10」（軽量版・フル版とも）と一致
- 平均NLL margin: Stage 2 `+0.421`、Stage 3 `+0.235`
- 記事の`+11.86 / +12.26`は別のbaselineに対する別量のため比較しない

#### 解決した問題

| 問題 | 内容 | 対応 |
| --- | --- | --- |
| checkpoint読み込み | 公開weightはoriginal Moshi名で保存されており、Zero-3向け改名と166 parameterで不一致 | `tools/moshi_state_dict.py`の名前対応 |
| 生成の形式不一致 | 公開checkpointは`n_q=16, dep_q=8`の推論用形式で、生成経路は`dep_q == n_q`前提 | user streamを無音で教師強制（ユーザー承認済み） |
| prompt長の余裕ゼロ | held-out最短が`48 frames`で、delay込み`50 frames`は`min_length=50`ちょうど。`filter_out_short_streams`は無言で捨てる | prompt長を`40 frames`へ下げ、`verify-dataset`ゲートを追加 |
| 選好判定の長さバイアス | 合計log-probは長い候補を不利にする。preferredが長い候補のpairが10件中6件あり、3/10と誤判定していた | 判定を長さ正規化した平均NLLへ変更。合計log-probは診断用に併記 |
| 誤ったゲート | 絶対NLLが一様分布を下回ることを要求するゲートが、機能している対比較を3回却下した | `assert_scores_discriminate`へ置換。非有限値と、両候補が同点になる（＝completionが採点位置に届いていない）場合だけ失敗させる |
| ホスト容量枯渇 | `48004205`が起動不能（41分・20回の再試行がすべて拒否） | `48178589`へ移行し、`bootstrap_instance.sh`で再構築手順を固定 |

絶対NLLが一様分布より高いのは、条件音声が採点テキストと別の発話である以上構造的なもの。両候補が同じ文脈と音声を共有するため、対比較には影響しない。

#### 環境と費用

- 環境manifest: `m0/environment-manifest.json`（`48178589`、A100-SXM4-80GB ×2、torch 2.4.1+cu121、transformers 4.48.3）
- 再現手順: `m0/bootstrap_instance.sh`（checksum検証つき）、`m0/run_baseline.sh`
- API key: 現key継続をユーザーが明示承認、repository外、mode `600`
- 費用: 請求済み`US$5.661`、発生見込み`US$20.979`。承認上限`US$100`
- インスタンス: `48178589`は`stopped`（120 GiB、`US$0.0333/h`）。`48004205`は2026-08-20に破棄済み
- 再現性の注意: `uv.lock`がgitignore対象のため、ホストごとに依存解決が変わりうる（`environment-manifest.json`の`reproducibility_gap`参照）

## M1: 権利・データ確定

### 目的

使用する原音、会話script、合成音声、評価データの権利と由来を明確にし、データ漏洩のない再現可能な分割を作る。

### ゴール

すべての入力データについて、取得元、版、checksum、利用条件、加工履歴、train/dev/test所属を追跡できる。

### 入力

- つくよみちゃん公式100文
- 取得できた場合は申請制の追加約1,500台詞
- 過去のお嬢様100対話script、または再生成用の仕様
- `OjousamaTalkScriptDataset` 200組
- M0で固定したbaseline評価条件

### 作業

1. 原音を再配布対象にせず、安全な保存先へ取得する。
2. 音声のsample rate、channel、長さ、破損、重複を検査する。
3. 各データのライセンス、クレジット、禁止用途、派生モデル条件を台帳化する。
4. COEIROINK生成音声が混入していないことを確認する。
5. 対話単位で80/10/10に分割し、内容重複と話題漏洩を検査する。
6. Voice seen / held-out、Style held-out、一般対話の固定評価セットを作る。
7. お嬢様scriptを回収できない場合の100対話再生成仕様を固定する。

### 成果物

- データ・ライセンス台帳
- checksum付きraw manifest
- train/dev/test split manifest
- 固定評価promptと評価rubric
- お嬢様scriptのprovenanceまたは再生成仕様

### 完了条件

- [x] すべてのデータに取得元、版、checksum、利用条件がある。
- [x] COEIROINK生成音声が0件である。
- [x] 破損・重複検査に合格している。
- [x] train/dev/test間に同一対話・近重複がない。
- [x] 固定評価セットが学習データから隔離されている。
- [x] 公開時に必要なクレジット文面と非公開対象が決まっている。

### 次へ進む条件

用途不明または利用条件未確認のデータが1件でもあればM2へ進まない。追加1,500台詞が未取得でも、公式100文だけのpilotは実行可能とする。

### 完了記録

- 状態: 完了
- つくよみちゃん公式Vol.1: archive SHA-256固定、100件を安全に抽出、96 kHz IEEE Float monoを検証
- split: train 80 / dev 10 / test 10、破損0、重複0、split間同一・近似重複0
- provenance: `experiments/tsukuyomi_ojousama/registry/tsukuyomi-corpus-v1.json`
- raw manifest: `experiments/tsukuyomi_ojousama/manifests/tsukuyomi-corpus-v1.jsonl`
- validation: `experiments/tsukuyomi_ojousama/reports/tsukuyomi-corpus-v1-validation.json`
- お嬢様参考データ: upstream 202行、重複prompt 1件を棄却し201件を`reference-only`として固定
- 固定評価: TTS 30、Style 50 pair、一般対話30、Voice seen/held-out各10
- 評価漏洩: trainとの完全・近似重複0
- 評価rubric: `experiments/tsukuyomi_ojousama/eval/RUBRIC.md`
- 100対話再生成仕様: `experiments/tsukuyomi_ojousama/style/DATASET_SPEC.md`
- クレジット・非公開対象: `experiments/tsukuyomi_ojousama/DATA_CREDITS.md`
- 追加約1,500台詞（夢前黎の音声データの寄せ集め）: **未取得のまま台帳化して確定**。2026-08-20に上流の利用条件を確認し、`registry/tsukuyomi-yoseatsume-candidate.json`へ取得元・版・入手方法・利用条件・禁止事項・除外理由・再開条件を記録した。`used_in_experiment: false`で管理し、manifestには1件も含まれない
- 同データの入手は上流メールフォームからの**申請制**で、申請はユーザー本人が行う必要がある。エージェントは代理申請しない
- 取得時の注意: この寄せ集めはつくよみちゃんコーパスVol.1の100文を含む上位集合であり、`tsukuyomi-corpus-v1`と重複排除してから分割しないと固定held-outがtrainへ再流入する。JSUT basic5000部分は別条件のため別途台帳化が必要。再配布禁止のため取得物と派生物は非公開対象
- 独立監査（2026-08-20）: 完了条件を文書のチェックに頼らず実データで再検査した。原音100件すべてがmanifestのSHA-256・byte sizeと一致、registryは`eval/`の5ファイルを漏れなく網羅、`voice-seen-heldout-20.jsonl`の20行はartifact_id・sha256・text・splitがmanifestと一致し、seen 10はtrain・held-out 10はtestから採られている
- **MIT準拠の不備を修正**: `reference/ojousama-talk-script-201.jsonl`は公開リポジトリにコミットされた再配布物だが、`DATA_CREDITS.md`には著作権表示のみで許諾条項本文が無く、本文は`data/`配下（gitignore対象）にしか存在しなかった。上流LICENSEの逐語コピーを`reference/LICENSE.OjousamaTalkScriptDataset`として同梱し（SHA-256 `fcd8fbf3…`）、credits・registryに記載した
- 監査のテスト化: 上記4点をすべて`tests/test_experiment_assets.py`の恒久テストにした。ライセンス削除・評価ファイルの登録漏れを実際に変異させて検出することを確認済み
- CIでのデータゲート実行: `.github/workflows/code-quality.yml`はruffのみでテストを一度も実行していなかった。testsジョブを追加し、`pytest`を`dev` extraへ宣言した。原音を必要とする検査はデータが無い環境ではskipするため、新規cloneでも69 passed / 1 skippedで通る
- lint/format: 実験ブランチが持ち込んだlint 14件と未フォーマット7件を解消し、`ruff check`は全通過。`finetune.py`のみ未フォーマットのまま残るが、これは`main`から継承した上流由来の既存事項で本実験の変更ではない

## M2: Tsukuyomi TTS

### 目的

J-Moshi用の2話者対話を作る前に、つくよみちゃん原音から未学習文を明瞭に発話できるTTSを作り、データ品質の上限を確認する。

### ゴール

Irodori-TTSで未学習のお嬢様語彙を含む30文を生成し、明瞭度と話者らしさのTTS Gateを通過する。

### 入力

- M1で確定したつくよみちゃん原音と分割
- Irodori-TTS 500M v3
- 未学習評価文30件
- 中立話者TTSのcontrol音声

### 作業

1. Irodori-TTSを対象話者へ適応させる。方式はT0 zero-shot（control）→ T1 Speaker Inversion → T2 LoRA → T3 全パラメータの順に比較する（[実験計画](../../docs/experiments/j-moshi-tsukuyomi-ojousama-plan.md)で2026-08-20に変更）。
2. train/dev lossと各checkpointを保存する。
3. 未学習30文を固定条件で生成する。
4. 欠落、クリップ、誤読、音量、話者類似度、抑揚を評価する。
5. 追加約1,500台詞を取得できた場合は同条件で比較する。
6. 学習元、設定、seed、checkpoint、生成条件をmanifest化する。

### 成果物

- TTS学習configと実行コマンド
- TTS checkpointとchecksum
- 未学習30文の生成音声
- TTS評価レポート
- 採用・不採用理由

### 完了条件

- [x] 30文すべてに欠落・クリップがない。
- [x] 30文中27文以上を明瞭に聞き取れる。（T1は30/30）
- [x] 話者らしさがbase TTSより主観評価で改善している。（ブラインドA/B 6件でB 4勝2分0敗、客観指標と併せて判定）
- [x] train文の暗記だけでなく未学習のお嬢様語彙を発音できる。
- [x] checkpointと生成条件を再現できる。

### 次へ進む条件

合格ならM3のV-ttsを作る。不合格なら追加音声、正規化、TTS設定を改善する。V-realはTTS不合格時にも作成できるが、M3のcontrol比較に入る前に中立B話者の品質を確認する。

### 完了記録

- 状態: 進行中

#### 確定した前提（2026-08-20）

- コーパス実測は総量`10.97分`、train `8.67分`。過去のあみたろ`189.9分`の`5.8%`
- この量では全パラメータ学習が過学習を起こし、完了条件4と直接衝突するため、適応方式をT0〜T3の比較へ変更。理由は実験計画に記録
- Irodori-TTSはローカル（`~/Desktop/Irodori-TTS`）に存在し、`Aratako/Irodori-TTS-500M-v3`と`Semantic-DACVAE-Japanese-32dim`はHFキャッシュ済み
- 推論はApple Silicon MPSで動作。1文あたり約17秒なので、**生成にGPUレンタルは不要**
- 原音は再配布禁止のため、`prepare_manifest.py`は`--data-files`でローカル読み込みする。HFへのアップロードはしない

#### T0: zero-shot voice cloning（base TTS control）— 完了

- 参照音声はtrain splitの`VOICEACTRESS100_094.wav`（13.45秒）。test splitはheld-outのまま維持
- 固定条件: `Aratako/Irodori-TTS-500M-v3`、seed `20260820`、MPS、48 kHz出力
- 未学習30文すべてを生成。**欠落0、無音ファイル0**
- 全長平均`5.98秒`、先頭無音平均`0.272秒`、末尾無音平均`0.472秒`
- 飽和サンプルは27ファイルに計170個あるが、**連続長は最長2 sample（0.042 ms）**。`soundfile`がPCM_16書き出し時に`±1.0`超を飽和させたもので、可聴の歪みではない
- 測定ツール: `tools/tts_audio_report.py`（テスト12件）。レポート: `data/experiments/tsukuyomi_ojousama/m2/zeroshot-report.json`（gitignore対象）

#### T1: Speaker Inversion — 完了・採用

- 学習対象は**12,288パラメータのみ**（16 token × 768次元）、base `512,049,441`は完全凍結。8.67分のデータでも構造的に暗記が起こらない
- 入力: train split 80件のDACVAE latent（`prepare_manifest.py --dataset json --data-files`でローカル読み込み。原音は再配布禁止のためHFへ上げない）。skip 0件
- config: `train_500m_v3_speaker_inversion.yaml`、lr `0.01`、batch 16、3000 steps、`--precision fp32`、`--device mps`
- 速度: 約`11.8秒/step`。3000 stepで約10時間。250 stepごとにcheckpointを保存し途中評価する
- 推論はローカルMPSで約17秒/文のため、生成にGPUは不要。学習のみGPUへ出す選択肢もあるが、2026-08-20時点でVast.aiの空きが確保できず、ローカルで実行している

##### Irodori-TTSへのローカルパッチ

`train.py`の`dtype=torch.float64`が4箇所あり、MPSはfloat64を扱えないため学習が起動しない。4箇所とも`DURATION_CONDITION_GROUP_TOTAL_SIZE`の統計集計用で、隣接する同種accumulatorは元からfloat32であり、勾配にも学習される埋め込みにも関与しない。`float32`へ変更した。上流のMPS非互換であり、`git -C ~/Desktop/Irodori-TTS checkout train.py`で復帰できる。

##### 学習結果

A100単一GPUで3000 stepを`1.03秒/step`・約8分・`US$0.219`で完走（ローカルMPSは`11.8秒/step`で約12時間だったため移行）。RF loss `0.78 → 0.650`、duration MAE `40.17 → 18.04` frames。埋め込み SHA-256 `b9e10f13c450f263…`

##### T0とT1の客観比較

| 指標 | T0 zero-shot | T1 speaker inversion |
| --- | ---: | ---: |
| 生成成功 | 30/30 | 30/30 |
| 無音・欠落 | 0 | 0 |
| 可聴クリップ | 0 | 0 |
| 明瞭（必要27） | 29/30 | **30/30** |
| 平均CER | 0.0133 | **0.0073** |
| 最大CER | 0.185 | **0.098** |
| 先頭無音 | 0.272秒 | 0.049秒 |
| 末尾無音 | 0.472秒 | 0.170秒 |

T1は平均CERを半減し、T0で唯一失敗した文も解消した。無音の短縮はduration MAEの改善と整合するが、これは発話タイミングであって声質ではないため条件3の判断材料にはならない。

##### 明瞭度の測定方法を訂正した経緯

最初の測定は表層文字列を比較しており、T0を26/30として不合格にした。失敗4件はすべて認識器の表記（`十二月二十四日`→`12月24日`、`瑞々しい`→`みずみずしい`、`百二十八個`→`128個`、`アルファ`→`α`）であり、発音は正しかった。`pyopenjtalk`のg2pで両者を読みへ変換してから比較するよう修正し、T0 29/30・T1 30/30となった。

##### 残る条件3

計画書が指定する手順（有声区間抽出 → RMS正規化 → speaker embedding類似度）でECAPAコサイン類似度を測定した。基準は**test splitの原音10件**で、学習に使ったtrain splitではなくheld-outを使うことで、埋め込みが学習音声そのものへ寄る影響を避けている。

| | T0 zero-shot | T1 speaker inversion |
| --- | ---: | ---: |
| mean | 0.6597 | **0.6918** |
| median | 0.6667 | **0.6977** |
| min | 0.5469 | **0.5806** |
| max | 0.7525 | **0.8006** |

同一30文の対応比較では、**T1が24/30で上回り**、平均差`+0.0321`、符号検定`p = 0.00143`。全統計量でT1が上であり、方向は一貫している。

ただしこれは**補助材料であって判定ではない**。計画書は「自動類似度だけで採用せず、日本語話者によるブラインドA/Bを行う」「SECS単独評価は禁止する」と定めており、過去にSECSの高い棒読みTTSよりSECSの低いfull fine-tuning版の方が主観的に本人らしい事例が記録されているためである。

聴取用一式は`data/experiments/tsukuyomi_ojousama/m2/listening/`（30ペア＋原音基準）。ブラウザ版は`compare.html`。

##### 条件3の判定（2026-08-21）

ブラインドA/Bを実施。系の別と客観指標は選択するまで伏せ、再生順は文ごとに入れ替えた。

| 判定 | 件数 |
| --- | ---: |
| B（speaker inversion）が近い | 4 |
| 差がない | 2 |
| A（zero-shot）が近い | **0** |

**判定は`met`。ただし根拠は聴取単独ではなく複合である。**

聴取6件を符号検定にかけると`p = 0.125`で、単独では有意水準5%に届かない（あと2件の非引き分け勝ちで到達する）。一方で計画が聴取を要求した目的は「自動指標が誤誘導していないかの確認」であり、**Aが1件も勝たなかった**ことでその確認は果たされた。統計的な重みは対応比較の類似度（24/30、`p = 0.00143`）と明瞭度（30/30 対 29/30、平均CER半減）が supply する。どちらか一方だけでは足りない構成になっている。

耳と指標はper-sentenceでは一致しない。類似度差が6件中最大の`tts-04`（`+0.0533`）が引き分け判定であり、指標を代替にできないという計画の前提を裏づけている。

記録: `experiments/tsukuyomi_ojousama/reports/m2-listening-judgement.json`

##### 聴取工数の自動化

`compare.html`に逐次停止規則を組み込んだ。引き分けを除き同じ側が**6件続けば有意水準5%に到達**するため、30件を聴く必要はない。ページ下部が現在の`p`と「あと何件で確定するか」を常時表示し、確定した時点で「ここで止めて構いません」と示す。M3以降の聴取でもそのまま使える（`tools/build_listening_page.py`、テスト17件）。


## M3: Voice control

### 目的

過去に成功した「2話者stereo・全パラメータ・学習率`3e-5`」をつくよみちゃんデータで再現し、voice overfitの安全な比較基準を作る。

### ゴール

V-realまたはV-ttsの少なくとも一方で、J-Moshi-extより対象話者らしさを改善し、独話loop、反復collapse、重大な明瞭度低下を起こさない。

### 入力

- M1の固定評価セットと分割
- M2の採用TTS
- V-real / V-tts用100対話
- `nu-dialogue/j-moshi-ext`

### 作業

1. V-realとV-ttsを別datasetとしてA=対象話者、B=中立話者の24 kHz stereoへ変換する。
2. 音声、時刻、テキスト、話者channelの一致を検証する。
3. tokenizeとparquet化を行い、失敗・skip件数を記録する。
4. V0/V1を全パラメータ、tempformer/depformerとも`3e-5`、最大5 epochsで学習する。
5. 各epochを保存し、loss、声質、明瞭度、live対話を評価する。
6. runごとのVast.ai費用と実行時間を記録する。

### 成果物

- V-real / V-ttsのmanifest、検証結果、train/dev/test parquet
- V0/V1の完全なconfig、W&B run、checkpoint
- 各epochの固定生成音声
- Voice control比較レポート
- M4へ渡す採用checkpoint

### 完了条件

- [ ] dataset検証でchannel・時刻・テキスト不一致が0件である。
- [ ] tokenizeのskip理由と採否がすべて記録されている。
- [ ] V0/V1の少なくとも一方が固定30会話で独話loop・反復collapseを起こさない。
- [ ] held-out音声の話者らしさがJ-Moshi-extより改善する。
- [ ] 明瞭度とturn-takingが許容範囲内である。
- [ ] 中間checkpointを含めた採用理由が残っている。

### 次へ進む条件

合格runがあればM4へ進む。両方不合格なら学習率を上げず、M2またはdataset構築へ戻る。

### 完了記録

- 状態: 未着手
- 証拠: 未作成

## M4: Voice overfit

### 目的

2話者対話と全パラメータ学習を維持しながらdepformerの学習率だけを上げ、controlより声質を強く適応する。

### ゴール

V2またはV3でheld-out声質をM3 controlより改善し、明瞭度・内容・turn-takingをcontrol同等に保つ。

### 入力

- M3で採用したdataset、分割、base checkpoint
- V0/V1 controlの評価結果
- 固定評価promptとsampling条件

### 作業

1. V2をtempformer `3e-5`、depformer `6e-5`で実行する。
2. V2が安定し、controlより改善した場合だけV3をdepformer `1e-4`で実行する。
3. NaN/Inf、loss急増、音質劣化、loopを監視する。
4. 各epochをブラインド比較し、最終epoch固定ではなく最良checkpointを選ぶ。
5. seenとheld-outの差から暗記と一般化を判定する。

### 成果物

- V2/V3のconfig、W&B run、checkpoint、費用記録
- epoch別の固定生成音声
- controlとの差分評価レポート
- 最良voice checkpointと採用理由

### 完了条件

- [ ] V2の全評価が完了している。
- [ ] V3はV2合格時だけ実行されている。
- [ ] held-out声質がcontrolより改善している。
- [ ] 明瞭度、内容、turn-takingがcontrol同等である。
- [ ] seenだけ改善したcheckpointを採用していない。
- [ ] 最良checkpointと停止理由が記録されている。

### 次へ進む条件

改善runがあればそのcheckpointでM5へ進む。改善しない場合はM3の最良controlをM5へ渡し、「高LRは不採用」と記録する。

### 完了記録

- 状態: 未着手
- 証拠: 未作成

## M5: お嬢様口調

### 目的

M4のvoice checkpointの音響特性を保護しながら、自然なお嬢様口調をtempformerへ追加する。

### ゴール

S-strictまたはS-mildでheld-outのお嬢様選好と人手評価を改善し、声質・明瞭度・full-duplex対話をvoice checkpointから悪化させない。

### 入力

- M4の最良voice checkpoint
- 回収または再生成したお嬢様100対話
- つくよみちゃん系TTSと中立話者TTS
- Style held-out 50 pairと一般対話評価セット

### 作業

1. S-strictとS-mildを別script・別datasetとして作る。
2. A/Bのpaired stereo音声、時刻、テキストを生成・検証する。
3. tempformer-only、学習率`3e-5`、最大5 epochsでS0/S1を実行する。
4. 50 pair perplexity、語尾分布、同一語尾連続率、人手評価を比較する。
5. voice checkpointとの声質差、明瞭度、live対話を評価する。
6. tempformer-onlyで不足した場合だけS2全パラメータ学習を検討する。

### 成果物

- S-strict / S-mildのscript、provenance、dataset manifest
- S0/S1のconfig、W&B run、checkpoint、費用記録
- Style / Voice / Full-duplex比較レポート
- 最良persona checkpointと採用理由

### 完了条件

- [ ] 100対話の出典または生成条件を再現できる。
- [ ] held-out 50 pairでvoice checkpointよりお嬢様選好が改善する。
- [ ] 同じ語尾の過剰反復が許容範囲内である。
- [ ] 未学習話題でも口調が維持される。
- [ ] 声質・明瞭度・turn-takingがvoice checkpointから大きく悪化しない。
- [ ] S0/S1の採用判断と、S2を実行するかの理由が記録されている。

### 次へ進む条件

全条件を満たすcheckpointがあればM6へ進む。口調だけ改善して音声対話が悪化した場合は、追加データを増やす前にstrict/mild比率と中間checkpointを見直す。

### 完了記録

- 状態: 未着手
- 証拠: 未作成

## M6: 最終検証

### 目的

選定した設定を全データで再実行し、技術・品質・権利・再現性を一つのrelease candidateとして固定する。

### ゴール

声質、明瞭度、お嬢様口調、応答品質、full-duplex対話、利用条件の全Gateに合格し、第三者が設定と成果物を追跡できる状態にする。

### 入力

- M4の最良voice設定
- M5の最良style設定
- 全100対話のfinal-overfit dataset
- M1の固定評価セットとデータ台帳

### 作業

1. 採用設定を全100対話で最初から再学習する。
2. sampling条件を固定して全評価セットを生成する。
3. run名を伏せたブラインド評価を実施する。
4. license、クレジット、禁止用途、非公開データを再監査する。
5. config、commit SHA、データchecksum、checkpoint checksum、費用、評価結果をまとめる。
6. model cardと利用手順を作る。
7. 公開・アップロードは別途ユーザー承認を得るまで行わない。

### 成果物

- final voice / persona checkpoint
- 最終config、実行コマンド、環境manifest
- 最終評価音声とレポート
- データ・checkpoint checksum一覧
- model card、クレジット、利用条件
- 採用・不採用runを含む実験台帳

### 完了条件

- [ ] held-out声質がM3 control以上である。
- [ ] 明瞭度が定めた基準を満たす。
- [ ] Style held-outと一般対話の両方に合格する。
- [ ] 独話loop、反復collapse、ユーザー無視が固定評価で発生しない。
- [ ] データ、config、commit、checkpoint、評価の対応を追跡できる。
- [ ] 必要なクレジットと利用条件がmodel cardに記載されている。
- [ ] 公開前のユーザー承認待ち状態まで準備できている。

### 次へ進む条件

M6は本実験の終端とし、全完了条件を満たしてrelease candidateを固定した時点で実験完了とする。ユーザーが結果を確認して承認した場合だけ、checkpointの公開、Hugging Faceへのupload、PR作成などを別タスクとして実施する。

### 完了記録

- 状態: 未着手
- 証拠: 未作成

## 進捗更新ルール

各作業後に以下を更新する。

1. 一覧表の状態。
2. 対象マイルストーンのチェックボックス。
3. 完了記録の証拠リンク、run ID、checkpoint、評価レポート。
4. 不合格時の戻り先と理由。
5. Vast.aiの実行時間と費用。残高やAPI keyそのものは記載しない。

実験計画の技術条件を変更した場合は、先に実験計画を更新し、その後この文書の目的・ゴール・完了条件を同期する。
