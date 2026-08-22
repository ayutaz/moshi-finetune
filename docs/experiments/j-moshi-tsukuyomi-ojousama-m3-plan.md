# M3 Voice control 実行計画

作成日: 2026-08-21

対象マイルストーン: [M3](./j-moshi-tsukuyomi-ojousama-milestones.md#m3-voice-control)

技術方針は[実験計画](./j-moshi-tsukuyomi-ojousama-plan.md)、完了判定は[マイルストーン文書](./j-moshi-tsukuyomi-ojousama-milestones.md)を正とする。
この文書は M3 の実行順序、ゲート、費用、および計画本文からの逸脱を管理する。

## 目的とゴール

過去のあみたろ実験で実際に動いた条件（2話者 stereo・全パラメータ・LR `3e-5`）をつくよみちゃんの
データで再現し、M4 の voice overfit が比較できる「壊れていない基準」を作る。

V-real（V0）と V-tts（V1）の少なくとも一方が、`nu-dialogue/j-moshi-ext` より対象話者らしさを
改善し、独話 loop・反復 collapse・重大な明瞭度低下を起こさないこと。

## 実測で確定した前提

計画を書く前に測ったもの。いずれもコードを読むだけでは確定しない。

| 事実 | 数値 | 記録 |
| --- | --- | --- |
| caption 毎回生成の話者Bは一人の話者にならない | pairwise ECAPA 0.4525（seed既定）/ 0.5176（seed固定） | [`m3-speaker-b-probe.json`](../../experiments/tsukuyomi_ojousama/reports/m3-speaker-b-probe.json) |
| 実人間一人の帯（校正基準） | 平均 0.6983 / 最小 0.5646 | 同上 |
| 1本を `--ref-wav` で凍結した話者B | 平均 0.7373 / 最小 0.7040 | 同上 |
| 話者A と凍結話者B の分離 | 0.157 | 同上 |
| Mimi はこの Mac で動く | CPU・MPS 両方、2s→25 frames | [`m3-local-compute-probe.json`](../../experiments/tsukuyomi_ojousama/reports/m3-local-compute-probe.json) |
| 160 対話のローカル tokenize | MPS 1.2 分 / CPU 2.2 分 | 同上 |

`--device` フラグは `036f5c7` で実装済み（`tools/worker_device.py`、テスト8件）。

## コードを読んで確定した前提

| 事実 | 値 | 根拠 |
| --- | --- | --- |
| dep_q=16 モデルのパラメータ数 | 8,371,408,896 | `tools/init_moshi_for_ft.py` |
| ZeRO-3 の `save_state` が書く量 | **12 bytes/param = 100.46 GB**（fp16 重みコピーは無い） | deepspeed 0.15.4 `stage3.py:2508-2521` |
| checkpoint ローテーション | **存在しない**（`--save_total_limit` 無し、accelerate の自動命名も無効） | `finetune.py` argparse 全36フラグ |
| 学習時の常駐メモリ | 16 bytes/param = 133.94 GB | ZeRO-3 + fp16 + AdamW、offload 無し |
| A100-SXM4-80GB の実容量 | 85.90 GB | `m0/environment-manifest.json` |
| ホスト RAM 要求 | **80 GB 以上**（`finetune.py:638-642` が rank ごとに fp32 で CPU ロード、`zero_to_fp32` も約67 GB） | 同上 |

**1× A100 は不可能。** 133.94 − 85.90 = 48.04 GB 不足し、活性化を1バイトも置く前に落ちる。

## 規模についての判断（2026-08-21）

V-real の話者A が 7.8 分（過去実験 189.9 分の 1/24）である点は、**この規模のまま進めると決定した**。

M3 の目的は M4 が比較する基準を作ることであって、声質の絶対到達点ではない。V-real と V-tts が
同じ規模で揃っていれば、量そのものの効果は寄せ集めコーパス取得後に M4 以降で測れる。
申請待ちで M3 を止める方がコストが高い。

**代償**: 完了条件4 で差が出なかった場合、「手法が効かない」のか「量が足りない」のか判別できない。
陰性結果はそのように報告し、手法の否定として記録しない。

## 逸脱と、その再開条件

| 逸脱 | 理由 | 再開条件 |
| --- | --- | --- |
| V-tts を 600〜800 turn ではなく V-real と同じ80対話240 turn で作る | 計画の 600-800 turn は「V-real も100対話」という前提で書かれていた。test 分割を守ると V-real は80対話が上限で、規模を揃えないと V0/V1 の差が「データ量の差」になり比較が成立しない | 寄せ集めコーパス取得後、両者を同時に増やす |
| 話者Bを1本だけ生成して凍結する | 計画は「中立TTS」としか書いておらず、発話ごと生成では一人の話者にならないことを実測した | VoiceDesign に話者を固定する API が入った場合 |
| V-real の話者A は train 80文のみ | 完了条件4が held-out 話者らしさを要求する。100文全部を使うと測定対象が消える | 寄せ集めコーパス取得後（重複排除が前提） |
| V の分割は 72 train / 8 dev / 0 test | 対話単位の test は作らない。held-out 評価はコーパス test 分割の音声で行うため | — |
| 対話は完全逐次・重なりなし・固定0.4秒間隔 | barge-in を学習分布に入れない。turn-taking の測定はできるが学習はしない | 実対話コーパス導入時 |
| V-tts の話者A は M2 の speaker inversion（T1） | 計画の全パラメータTTS fine-tune は M2 で不採用。T1 が採用済み | — |

## 費用・構成

| 項目 | 値 |
| --- | --- |
| GPU | **2× A100-SXM4-80GB**（1ノード） |
| ホスト RAM | 85 GiB 以上（80 GiB 未満は失格） |
| ディスク | **900 GB**（`--disk` は create 後に変更できない） |
| 計画費用 | **US$20.28**（6.60 時間） |
| 予備 | US$5.35（1 run 再実行分） |
| M3 合計 | **US$25.63** / 枠 US$30 |
| 累計見込み | US$46.83 / 上限 US$100 |
| 打ち切り線 | create から 9.0 時間 = US$27.53 |

**停止インスタンスが最大の予算破壊要因。** 900 GB を停止状態で放置すると **US$10.00/日**。
M3 は当日中に destroy する。停止して翌日に持ち越す場合は台帳に記録し、再度 preflight を通す。

### ディスクが900 GB必要な理由

草案の 300 GB は **V0 の2本目の checkpoint 書き込み途中で死ぬ**。実効279 GB から固定分90.14 GB を
引いた188.9 GB に対し、2本目は200.91 GB を要求する。

`reclaim_checkpoints.sh` が学習と並行して閉じた ZeRO state を bf16 に変換して削除するが、
watcher が死んだ場合の最悪ケースを吸収できる容量が要る。900 GB はその余裕を含む。

## ステップ

### 第1部: ローカル準備（無課金、ステップ1〜18）— **完了（2026-08-22）**

課金が始まる前にすべての測定器とデータを完成させる。**測れないものを作る run は、作り直す run である。**

全18ステップ完了。GPU課金は発生していない（累計 US$25.638 はすべて M0・M2 分）。

| 通過したゲート | 結果 | 記録 |
| --- | --- | --- |
| 条件1 dataset 一致 | 160ペアで9種類の不一致がすべて0 | [`m3-dataset-agreement.json`](../../experiments/tsukuyomi_ojousama/reports/m3-dataset-agreement.json) |
| 条件2 tokenize 台帳 | skip 0、dropped 0、160ペア全数説明 | [`m3-tokenize-report.json`](../../experiments/tsukuyomi_ojousama/reports/m3-tokenize-report.json) |
| 話者B一貫性 | 平均0.8390、最悪ペア0.6542（実人間の平均0.6983に近い） | [`m3-speaker-b-drift-gate.json`](../../experiments/tsukuyomi_ojousama/reports/m3-speaker-b-drift-gate.json) |
| script 検証 | A文が train 80文と集合一致、eval 重複0 | [`m3-script-validation.json`](../../experiments/tsukuyomi_ojousama/reports/m3-script-validation.json) |
| manifest | 両者 72/8、重複0、破損0 | [`m3-manifest-validation.json`](../../experiments/tsukuyomi_ojousama/reports/m3-manifest-validation.json) |
| prompt set | 3種。general30 は生成区間内に37〜80フレームのユーザー音声 | [`m3-prompt-sets.json`](../../experiments/tsukuyomi_ojousama/reports/m3-prompt-sets.json) |

実データから確定した実行パラメータ:

| 項目 | 値 |
| --- | --- |
| train / dev | 72 / 8（両データセットで一致） |
| streams per example | 17 |
| global batch size | 8 |
| **S（steps per epoch）** | **9** |
| total steps | 45 |
| checkpoint | 9/18/27/36/45 の5本 = 502 GB |
| 起動時 assertion | `Num examples 72` / `batch 8` / `steps 45`。違えば即 kill |

課金前に見つけて直した欠陥（いずれも GPU 上なら金を失っていた）:

1. 話者Bが発話ごと生成では一人にならない（0.45〜0.52 < 実人間の下限0.565）
2. 全生成音声に SilentCipher の透かしが入っていた
3. `prepare_dataset` が不一致でも終了コード0を返す
4. `dialogue_id` にローカル絶対パスが焼き込まれる
5. parquet の行順が `os.listdir` 依存で非決定的
6. sidecar と wav の不整合で1ターンが消える
7. `tokenize_text` の既定 tokenizer が英語向けで全対話 skip
8. torch / torchaudio のバージョン不整合で `tokenize_audio` が起動しない
9. チャンネル判定が全体RMS比で全80対話が不合格（`B→A→B` でBが2ターン）
10. テキスト一致が表層比較で数詞展開（`1931`→`千九百三十一`）を不一致と判定
11. prompt が assistant 側しか作れず、生成区間でユーザーが常に無音
12. 既存の重複検出が逐語再生を見落とす（Jaccard 0.3967）
13. 停止インスタンスのディスク課金 US$4.44 が台帳から漏れていた

| # | 作業 | ゲート |
| --- | --- | --- |
| 1 | `m3/DATASET_SPEC.md` に6条件の証明ファイル・逸脱・閾値を先に書く | 6条件それぞれに報告パスとステップ番号。閾値がデータ無しに決められない項目は「校正で決める」と明記し、空欄にしない |
| 2 | `tests/test_experiment_assets.py` の silent-pass を塞ぐ。manifest を glob 化し、コーパス dev/test 由来の行が train に入ったら落ちる assertion を追加 | 全件green。leaky fixture に対して**新 assertion が落ちる**こと |
| 3 | 課金GPU上で失敗する／黙って何も出さない7ツールを修正 | 全件green。`prepare_dataset` が stem 不一致で**非ゼロ終了**しリポジトリ直下にゴミを残さない |
| 4 | 新規ソース8件の registry 登録と DATA_CREDITS 追記。**VoiceDesign のライセンスを読む** | 全 registry に `source_url`/`source_version`/ライセンス。dataset_id 重複なし |
| 5 | collapse 検出器を書き、M0 の既払い生成20件で校正 | 20件分類完了。合成陽性（4-gram×5回）で発火し、陰性で沈黙 |
| 6 | turn-taking 検出器を書く | M0 の20 wav で monologue を検出できること。**できなければ不在を証明できない** |
| 7 | paired sign test と channel splitter。M2 の確定値で自己検証 | M2 の `pairs 30 / higher_on 24 / mean_delta 0.03213770653093994` を**桁まで再現** |
| 8 | 暗記検出器。containment + 正規化部分文字列（Jaccard では見えない） | 敵対ケースで `_near_duplicate` が False・Jaccard 0.375 の一方、containment 1.0 |
| 9 | 80対話 script の作成と検証、共有 split map の確定 | A の文集合が train 80文と**集合として一致**。eval 重複0。最短200 frames |
| 10 | 話者Bを1本だけ生成して凍結。watermark 無効を確認 | A との ECAPA 平均 **0.30未満**。`silentcipher` が import できないこと |
| 11 | 全ターンをローカル生成し、話者B変動と長さ整合をゲート | プール191ターンで平均・中央値 **0.65以上**、5パーセンタイル **0.55以上**、0.50未満のペア0 |
| 12 | stereo 対話と語単位 transcript を組み立て、**条件1のゲート**を通す | 160ペアで channel/時刻/テキスト不一致・非stereo・SR誤り・長さ不足がすべて0 |
| 13 | manifest 2本を、コーパス dev/test 混入が構造的に起きない規則で構築 | `{train:72, dev:8, test:0}`、重複0、derivation が非空 |
| 14 | ローカル tokenize し、**train と dev の parquet を別々に**作る | npz 数 == wav 数 == transcript 数 == 80。skip 0件。skip は吸収せず**データ側を直す** |
| 15 | steps-per-epoch S をオフラインで確定し、smoke 用 parquet を切る | `num_streams == [17]`。行数72が**両データセットで一致**（倍なら `--moshi_speakers` が2値を拾っている） |
| 16 | held-out 10 と **seen 10**（暗記検出用）の prompt set を M0 の5段階全部で構築 | 各10行、`--min-frames 165`。held-out の順序が M0 の記録と一致 |
| 17 | general 30 の prompt set を、**生成区間の内側にユーザー音声が入る**ように構築。barge-in は**別ディレクトリ**に置く | 30行。全 prompt で frames 39..287 に**25 frames以上**のユーザー有声。`general30-user-wavs/` は**ちょうど30本** |
| 18 | 予算 preflight、台帳のスキーマ修正、レンタル判断の記録、**M3 ツリーを origin に push** | `experiment_budget` が 9.0h で allow、12.0h で**非ゼロ終了**。push 済み HEAD を記録 |

### 第2部: GPU（課金、ステップ19〜30）

1セッションで control → V0 → V1 → live をすべて行い、**当日中に destroy する**。

| # | 作業 | $ | ゲート |
| --- | --- | ---: | --- |
| 19 | 2× A100 / `--disk 900` を借り、j-moshi-ext だけ bootstrap | 1.38 | `actual_status == running` を**断定ではなく検証**。HEAD がローカルと一致。`MemAvailable >= 80 GB` |
| 20 | base モデル2種を構築、凍結データを upload、実測 S を `--save_steps` に固定 | 1.07 | 学習用 `dep_q 16`、control 用 `dep_q 8` |
| 21 | **一度も実行されたことがない**学習経路を smoke test し、実測 checkpoint でディスクを検算 | 0.92 | 実測 checkpoint サイズ × 想定本数 + 固定分 < 実効容量 |
| 22 | **両学習の前に** J-Moshi-ext control を3 prompt set で生成 | 0.76 | 生成 `.npy` が 10 / 10 / 30 ちょうど |
| 23 | V0（V-real）学習。reclaim watcher を**起動assertion通過後に**並走 | 2.14 | 起動ログが `Num examples 72` / `batch 8` / `steps 45`。違えば即kill |
| 24 | V0 の変換残りと全5 epoch の生成 | 3.21 | export サイズは**等値ではなく範囲**（safetensors ヘッダ分） |
| 25 | V1 起動前に ZeRO state が残っていないことを証明 | 0.15 | 残存0、空き容量が2本目の要求を上回る |
| 26 | V1（V-tts）学習。同一 bootstrap・同一依存 | 2.14 | V0 と同じ 72 / 8 / 45。違えば**両データセットが揃っていない**ので比較無効 |
| 27 | V1 の変換残りと全5 epoch の生成 | 3.21 | 同上 |
| 28 | **live full-duplex パス**を破棄前に実行 | 2.14 | probe の rows 9-16 が `generate.py` と一致 |
| 29 | checksum 検証つき export（大きいものを最後に） | 1.47 | 全ファイルの sha256 が一致してから次へ |
| 30 | destroy し、台帳を精算 | 0.15 | destroy 前に export 検証済みであること |

> **注意**: 自動生成された手順に、本実験が作成していないインスタンス `48178589` / `48187958` を
> `-y` で破棄するコマンドが含まれていた。**この計画から除去した。** 他プロジェクトのインスタンスは
> 本実験が判断してよい対象ではない。

### 第3部: ローカル採点（無課金、ステップ31〜37）

**インスタンス破棄後に実行する。** 完了条件3・4・5・6 の証拠はここで作られる。

| # | 作業 | 証明する条件 |
| --- | --- | --- |
| 31 | 全 `generated_tokens` を `--device cpu` で decode、channel 分離、テキストストリームを decode | 前提 |
| 32 | collapse 検出器を33ツリー全部に適用 | 条件3 |
| 33 | speaker_similarity の paired 比較（held-out と seen） | 条件4 |
| 34 | 暗記判定（seen vs held-out + containment） | 条件4 |
| 35 | `success_criteria` 採点と CER | 条件5 |
| 36 | 盲検聴取（逐次停止規則つき） | 条件4の裏づけ |
| 37 | `m3-voice-control-gate.json` を組み立て、マイルストーン文書に記録 | 条件6 |

## 完了条件と、事前登録した合格基準

**数値を見る前に決める。** `CLAUDE.md` は、根拠のない gate が動いている指標を3回却下した事例を記録している。

| # | 条件 | 証明ファイル | 合格基準 |
| --- | --- | --- | --- |
| 1 | dataset 検証で不一致0 | `m3-dataset-agreement.json` | 160ペアで9種類の不一致カウントがすべて0 |
| 2 | tokenize の skip がすべて記録 | `m3-tokenize-report.json` | `verdict_count == 160`、`unaccounted == 0` |
| 3 | 独話 loop・反復 collapse なし | `m3-collapse.json` | 少なくとも一方の run の採用 epoch で、30会話全部について `monologue_loop_count == 0` かつ `exact_repeat_collapse_count == 0` |
| 4 | held-out 話者らしさが改善 | `m3-speaker-likeness.json` | `mean_delta > 0` かつ 10件中8件以上で上回り、`mean_delta >= +0.02`。**加えて**暗記判定が `memorisation` でないこと |
| 5 | 明瞭度と turn-taking が許容範囲 | `m3-intelligibility.json` / `m3-turn-taking.json` | 盲検明瞭度が control 比 0.5 点以内、general 成功率が control 比10ポイント以内 |
| 6 | 中間 checkpoint の採用理由 | `m3-voice-control-gate.json` | 10行（2 run × 5 epoch）全部に loss・collapse・likeness・turn-taking が揃う |

条件4の sign test p は**参考値**として記録する。held-out 10件では 9/1 でようやく p=0.021 に届き、
8/2 は p=0.109 にしかならないため、**p 単独では判定しない**。効果量・方向・聴取の3点で判断する。

## 中止・保留条件

- ステップ4で VoiceDesign のライセンスが派生音声を制限していた場合、**M3 を止めて確認を取る**。
  話者Bは両データセットの全対話の片チャンネルを占めるため、これは権利判断であって技術判断ではない。
- ステップ10でどの caption も 0.30 分離を通らない場合、**bulk 生成に進まない**。
- ステップ11で話者B変動の中央値が 0.65 を割った場合、**tokenize せずに話者Bを作り直す**。
- 起動ログが `72 / 8 / 45` と違った場合、**即 kill**。データセットが想定と違う。
- ディスク850 GB 以上・RAM 85 GiB 以上の offer が無い場合、**その日は開始しない**。
  `--disk` は create 後に変更できず、300 GB では V0 の途中で死ぬ。
- create から 9.0 時間で打ち切る（US$27.53）。

## この M3 が establish しないこと

- **V-real と V-tts は、この設計でできる限り揃えたが完全ではない。** script・split・話者B音声・base・依存は
  同一だが、話者A の音声だけが違う。それが狙いである一方、TTS 側の音響特性の差は残る。
- **V0 と V1 の eval loss は互いに比較できない。** 各 run は自分の dev 音声で評価しており、
  V-tts の dev 音声は TTS 由来である。同じ列に並べてはならない。
- **条件4で差が出なかった場合、その結果は曖昧である。** train 分割は話者A の音声 8.67 分、
  M3 が使うのは72文で約7.8分。過去実験の189.9分の 1/24 である。
  差が出ないことは「手法が効かない」ことも「データが足りない」ことも意味しうる。
- **明瞭度は通常の意味の明瞭度ではない。** 生成対話に正解テキストが無いため、CER は
  Moshi 自身の decode したテキストに対してしか測れない。自己参照であり、
  反復するモデルは低い CER を出しうる。
- **turn-taking は重なりのない学習分布の上で測っている。** barge-in 追随は測れるが学習していない。
- **collapse の閾値は M0 の退化ケースで校正している。** M0 の B チャンネルは無音であり、
  健全な control ではない。閾値は「崩壊の署名」に対する校正であって「健全さ」の校正ではない。
- **live 対話の証拠は薄い。** 自動 probe 30会話 + barge-in 10件 + 人手10分程度。
  重大な崩壊は捕まえられるが、細かな品質差は判定できない。
- **ステップ29で export しなかったものはインスタンスと一緒に消える。** これは受け入れる。
- M3 は口調（M5）、公開可否、V2/V3 について**何も establish しない**。
