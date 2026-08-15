# GPT Live 型 LLM-jp-Moshi / J-Moshi エージェントの調査・設計メモ

更新日: 2026-08-16<br>
ステータス: 調査・設計段階。モデル／学習コードの変更は未着手。<br>
作業ブランチ: `research/j-moshi-tool-token-finetuning`

## 1. 結論

目標が次のすべてを満たす日本語の音声エージェントであるなら、**商用利用可能な `LLM-jp-Moshi-v1` を音声フロントエンドに採用し、ツール発火用の制御トークンを追加学習する**方針を第一候補とする。従来の `J-Moshi` は、実装・学習の参照先および研究用ベースラインとして扱う。

- 低遅延の full-duplex 音声会話、相槌、割り込み対応
- 権利処理済みの特定話者らしい声・話し方
- 「最新情報を調べる」「社内資料を探す」「重い推論をする」ことを会話中に自動判断
- 検索／RAG／バックエンド LLM の完了を待つ間も、自然に会話を継続

この方針は、[MoshiRAG](https://github.com/kyutai-labs/moshi-rag) の「モデルが RAG 発火トークンを出し、非同期取得結果を条件として戻す」実装を最短の出発点にする。より高度な並列知能統合は [KAME](https://arxiv.org/pdf/2510.02327) の oracle stream を後続の拡張候補とする。

`J-Moshi` の公開重みは **CC BY-NC 4.0** であり商用サービスには使えない。一方、LLM-jp が 2026 年 2 月に公開した [`LLM-jp-Moshi-v1`](https://huggingface.co/llm-jp/llm-jp-moshi-v1) は **Apache License 2.0** であり、商用利用を含む今回の出発点にできる。NII は同モデルについて、J-CHAT での事前学習と LLM-jp-Zoom1 での追加学習により構築され、J-Moshi より自然性・意味的適切性が改善したと報告している。[NII の公開告知](https://www.nii.ac.jp/news/release/2026/0225.html)

## 2. 目標と対象外

### 目標

ユーザーが「今週末の東京の天気を調べて」と話したとき、会話を止めずに「調べますね」と短く応じ、検索結果を受け取ると「土曜日は…」と続けられる音声エージェントを作る。これは、会話を維持したまま背後のより重い処理へ委譲する [GPT Live の考え方](https://openai.com/ja-JP/index/introducing-gpt-live/) に近い体験である。

### 対象外

- 一般に配布する任意話者の one-shot voice cloning
- まずから Web 検索・社内 RAG・予約・決済など多数のツールを同時に学習すること
- KAME の oracle stream を初期実装から再現すること
- 無許諾の人物音声や Web 上の音声を学習データに使うこと

## 3. 採用判断

### 推奨: LLM-jp-Moshi-v1 + 制御トークン + 非同期バックエンド

LLM-jp-Moshi-v1 を、即時性・会話性・声を担うフロントエンドとする。検索や長い推論は、モデル自身に完結させず外部サービスへ非同期に委譲する。

```text
マイク
  │
  ▼
LLM-jp-Moshi-v1 ── 可聴応答 ──────────────────► スピーカー
  │         「調べますね」
  │
  └── 非可聴の制御トークン <TOOL_SEARCH>
                 │
                 ▼
       セッション・オーケストレータ
          ├─ Web search
          ├─ 社内 RAG
          └─ バックエンド LLM
                 │
                 ▼
       参照テキスト／回答候補を Moshi へ注入
                 │
                 ▼
         「確認できた情報では…」と会話を継続
```

この分離では、Moshi は「いつ待つか」「どの種類の支援を求めるか」「待機中に何を発話するか」を担当し、検索品質・アクセス制御・ツール実行・長文推論はバックエンドで担当する。

### VoiceChat を第一候補にしない理由

[NVIDIA NemotronLabs VoiceChat](https://github.com/NVIDIA-NeMo/Speech/tree/nemotron-labs-voicechat) は、英語の end-to-end full-duplex 音声会話と tool calling を既に備える。ツール発火だけなら魅力的である。しかし本件では「特定の声で話す」が必須であり、公開済み VoiceChat チェックポイントは単一固定音声で、公式が voice cloning 非対応と明記している。

VoiceChat はカスケード型ではない。内部に音声認識、言語モデル、内部 EARTTS 音声生成器が統合されている。したがって、外部 TTS の話者埋め込みを後段に差し替えるだけでは解決しない。ソースには音声 prompt に対応した [EARTTS 学習コード](https://github.com/NVIDIA-NeMo/Speech/blob/nemotron-labs-voicechat/examples/speechlm2/duplex_eartts_train.py) や、特定話者 latent を checkpoint に登録する [voice-lock スクリプト](https://github.com/NVIDIA-NeMo/Speech/blob/nemotron-labs-voicechat/examples/speechlm2/nemotron-labs-voicechat_infer_voice_lock.py) がある。

ただし、これは公開 11B に少量音声の LoRA を当てる完成済み手順ではない。独自 TTS／STT／RNNT のチェックポイントを与えて再結合する [checkpoint 結合スクリプト](https://github.com/NVIDIA-NeMo/Speech/blob/nemotron-labs-voicechat/examples/speechlm2/combine_ckpt_conda.sh) が示すとおり、内部構成を理解して再構成・検証する研究開発になる。また voice-lock は、登録済み話者以外の任意 WAV による cloning を無効にするため audio-prompt projection を再初期化する実装を含む。

このため、VoiceChat は「英語中心で、固定声は気にしないか、相当のモデル開発資源がある」場合の候補であり、本件の初期選択にはしない。少なくとも調査時点では、公開 VoiceChat を特定話者へ安定して適応したコミュニティの再現可能な学習レシピ／派生チェックポイントは確認できなかった。

### ライセンスと運用条件の比較

VoiceChat のソースコードは Apache-2.0、公開重みは [OpenMDW 1.1](https://github.com/NVIDIA-NeMo/Speech/blob/nemotron-labs-voicechat/LICENSE_OpenMDW-1.1) で提供されている。OpenMDW 1.1 は改変・配布・商用利用を許すライセンスだが、再配布時のライセンス・通知の維持、特許・権利侵害に関する条項、無保証、第三者権利の確認責任がある。したがって「使いやすい」ことと「どの用途にも自動的に安全」なことは同義ではなく、プロダクト公開前の法務確認は必要である。

`J-Moshi` は本リポジトリのコード自体が Apache-2.0 でも、公開モデル重みは CC BY-NC 4.0 である。これに対し、`LLM-jp-Moshi-v1` はモデルカードと公式リポジトリの双方で Apache-2.0 と明記されている。[モデルカード](https://huggingface.co/llm-jp/llm-jp-moshi-v1) [公式リポジトリ](https://github.com/llm-jp/llm-jp-moshi)

したがって、商用プロダクトでは **J-Moshi の重みではなく LLM-jp-Moshi-v1 を明示して取得・追加学習する**。ただし Apache-2.0 であることは、学習に使う対象話者の音声、追加データ、Web/RAG のコンテンツの権利処理を不要にするものではない。

## 4. 既存リポジトリで確認した実装状況

このリポジトリは Moshi/J-Moshi の追加学習用実装であり、現状でも会話音声の追加学習経路を持つ。LLM-jp-Moshi-v1 も Moshi 系の checkpoint だが、ここでそのまま追加学習できるかは、Moshi のバージョン、checkpoint 読み込み、tokenizer、モデル形状を最初に照合してから判断する。

- [`README-ja.md`](../README-ja.md) は、J-Moshi が日本語会話データで構築されたこと、学習が Accelerate + DeepSpeed を前提とすることを説明している。
- [`tools/prepare_dataset.py`](../tools/prepare_dataset.py) は、テキストと音声 codec token を会話ストリームとしてまとめる。
- [`tools/init_moshi_for_ft.py`](../tools/init_moshi_for_ft.py) は、追加学習用モデルの初期化と、必要に応じたユーザー側音声ストリームの拡張を行う。
- [`finetune.py`](../finetune.py) は、text stream と audio stream のクロスエントロピー損失を計算している。`text_linear` が text vocab の logits を出しており、現状にはツール専用ヘッド、ツール実行器、結果注入器はない。

従って、対象話者の会話音声で「声・話し方・相槌」を適応する土台はある一方、制御トークン、非同期ツール実行、参照結果の注入は本ブランチで新規に設計・実装する対象である。

J-Moshi 構築時には 128 枚の V100 32 GB を約 36 時間利用したとリポジトリ README にある。これは基盤モデル構築時の数字であり、本件の追加学習コストを意味しない。一方で、追加学習も DeepSpeed 前提であり、単一 GPU で気軽に完結すると仮定してはいけない。LLM-jp-Moshi-v1 の公式モデルカードは、推論に **24 GB 以上の VRAM を持つ Linux GPU マシン**を要件としている。[LLM-jp-Moshi-v1 モデルカード](https://huggingface.co/llm-jp/llm-jp-moshi-v1)

比較として VoiceChat の公式デプロイ前提は NVIDIA GPU 80 GB 以上、Linux x86_64 などであり、初期検証の環境制約は重い。[VoiceChat prerequisites](https://github.com/NVIDIA-NeMo/Speech/blob/nemotron-labs-voicechat/docs/source/voicechat_realtime_instructions/prerequisites.md) ただし、Moshi の 24 GB は推論目安、VoiceChat の 80 GB は公式デプロイ前提であり、追加学習の必要 GPU を同列に比較するものではない。

## 5. 制御トークンの設計

### 5.1 何を学習するか

モデルの text stream に、画面や音声に出さない制御記号を出力させる。初期のツール集合は三つに限定する。

| 制御トークン | 意味 | 実行主体 |
| --- | --- | --- |
| `<TOOL_SEARCH>` | 変化し得る公開情報を Web 検索する | search backend |
| `<TOOL_RAG>` | 許可された文書集合を検索する | RAG backend |
| `<TOOL_LLM>` | より長い推論・コード・構造化処理を依頼する | backend LLM |

たとえば、教師例は次のように設計する。

```text
入力（会話文脈）
  ユーザー: 「今週末の東京の天気は？」

出力 text stream（概念表現）
  <TOOL_SEARCH> 調べますね。

出力 audio stream
  「調べますね」に対応する対象話者の音声 codec token

外部結果注入後
  「土曜日は晴れで、最高気温は…」に対応する text/audio token
```

発火トークンだけでは検索クエリや権限は決めない。初期 PoC では、オーケストレータが直近の ASR／会話履歴をバックエンド LLM へ渡して検索クエリを作る。この方が、Moshi に JSON 引数・権限判断・複雑な tool schema を同時に学習させるより安全である。

### 5.2 「新しい token を追加する」ことの意味

ここでいう token は単なる文字列ではなく、text tokenizer が一意に扱える制御 token を意味する。`[[SEARCH]]` のような既存文字列を並べる試作はできるが、複数 token に分割される、通常発話との誤検出が起きる、将来のツール追加に弱いという欠点がある。

本実装では次のどちらかを選ぶ。

1. **既存の未使用 special token を予約する**: tokenizer、埋め込み、出力語彙を増やさない。最短だが、対象 token が本当に未使用で、推論・デコード・学習で安全に扱えることを先に検証する必要がある。
2. **新しい special token を追加する**: `<TOOL_SEARCH>` などを tokenizer に追加し、語彙サイズ、`text_emb`、`depformer_text_emb`、`text_linear` とその checkpoint の整合を更新する。追加行は初期化後に学習する。実装量は増えるが、意味と監査性が明確で本番向きである。

MoshiRAG は RAG 用に専用の token id を監視するため、後者または安全に確保した専用 token という考え方の先例になる。まずは special token の追加可否と、LLM-jp-Moshi-v1 の SentencePiece tokenizer に予約済み token があるかを小さな検証で確認する。未確認の token id をハードコードしてはならない。

### 5.3 学習時に必要な変更

現状の `finetune.py` は既存 text vocab に対する損失しか持たない。専用 token を追加する場合は少なくとも次を実装・テストする。

1. tokenizer が制御 token を単一 id で encode/decode できるようにする。
2. text vocab のサイズを増やし、入力 embedding と text 出力線形層を安全に拡張する。
3. 新規行の初期化方法を決める（ランダム初期化、既存 special token 近傍からの初期化など）。
4. データ前処理で、ツール発火対象の教師 text stream に制御 token を挿入する。
5. 推論サーバーで stream 中の token を一度だけ検出し、UI の文字起こし・外部へ出すテキストから除外する。
6. token 発火時に、可聴音声が「調べますね」のような自然な待機応答になるデータを含める。
7. 推論時の stop、padding、BOS/EOS、遅延 token と制御 token の相互作用を回帰テストする。

重要なのは、制御 token を見えないようにフィルタするだけでは不十分な点である。text/audio を同時生成するモデルなので、制御 token を出す瞬間に音声が不自然にならないことをデータと評価で保証する必要がある。

## 6. バックエンドとの統合方式

### 6.1 MoshiRAG を初期実装の基準にする

[MoshiRAG](https://github.com/kyutai-labs/moshi-rag) は、Moshi の内部テキストと streaming ASR を文脈として、RAG 発火 token を検出する。検索は非同期で実行され、得た reference text をストリーミング conditioner として推論へ注入する。これは本件の `<TOOL_RAG>` に直接対応する。

初期実装では、MoshiRAG の仕組みを一般化し、token と backend を対応付ける。

```text
<TOOL_SEARCH> -> Web 検索 -> 要約済み参照テキスト
<TOOL_RAG>    -> ACL 付き文書検索 -> 根拠テキストと出典
<TOOL_LLM>    -> 大規模 LLM -> 回答案または構造化結果
```

バックエンドが失敗・タイムアウトした場合も、Moshi の会話を壊さない。「今は確認できませんでした」「別の方法で確認します」のような失敗応答を、通常の会話状態へ戻すイベントとして扱う。

### 6.2 KAME は第2段階以降

KAME は Moshi に、外部 LLM の結果を表す追加の text stream（oracle stream）を持たせ、入力の進行に合わせて段階的に外部知能を供給する。検索結果を一度だけ注入する MoshiRAG より、会話と重い推論をより密に並列化できる。

ただし、追加 stream、データ合成、時間整列、学習損失を含むため、既存 J-Moshi 追加学習コードへの改変範囲は大きい。初期プロダクトでは MoshiRAG 型の reference conditioner で価値を検証し、待ち時間・複雑な推論・長い tool cycle がボトルネックになってから KAME 型を検討する。

### 6.3 セッション状態と中断

full-duplex では、ユーザーが結果を待たずに話し始める。オーケストレータには少なくとも次の状態管理が必要である。

- 発火ごとの `tool_call_id`、開始時点の会話 revision、期限を保持する。
- ユーザーの訂正・割り込みで、未開始の検索をキャンセルするか、結果を stale として破棄する。
- 結果は開始時点より古い会話へ注入しない。
- 同じ制御 token が連続生成されても、同一意図で多重実行しない（debounce / idempotency）。
- RAG 結果には文書 ID・アクセス制御・出典を保持し、回答だけを盲目的に注入しない。

## 7. 学習データ設計

### 7.1 特定話者の会話データ

対象話者本人または権利者から、学習・サービス利用・派生モデル利用に必要な明示許諾を取得する。音声だけではなく、書き起こし、話者の役割、ターン境界、相槌・重なり発話を可能な限り保持する。声質だけを狙うなら単独発話 TTS データも役立つが、full-duplex の自然な間・割り込みは会話データで評価する。

「数分の録音だけで完全な voice clone」を前提にしない。必要量は収録品質・話者の多様性・求める類似度・ベースモデルとの距離に依存するため、まず小規模データで類似度と会話品質の劣化を測る。

### 7.2 ツール発火データ

各例には少なくとも以下を持たせる。

| 項目 | 内容 |
| --- | --- |
| 会話文脈 | ユーザー音声／テキスト、直前ターン、必要なら時刻・地域 |
| 発火ラベル | `NONE`、`SEARCH`、`RAG`、`LLM` |
| 可聴応答 | 発火時の短い自然な応答、または即答 |
| 制御 token | 対応する special token（`NONE` には出さない） |
| ツール結果 | 根拠テキスト、出典、失敗・タイムアウトを含む |
| 継続応答 | 結果を利用した最終 text/audio 応答 |

特に `NONE` の負例が重要である。「今日の天気は？」と「明日の東京の天気は？」、「資料を探して」と「さっき説明した資料を要約して」のような近い例を含め、不要な検索発火を抑える。

Web の時事結果をそのまま固定正解として大量学習させるのではなく、発火判断と結果利用の会話を主に学習する。検索結果の事実性は実行時の取得と出典提示で担保する。

### 7.3 データ分割と安全性

- 話者・会話セッション・意図が train/validation/test をまたがないように分ける。
- ツール発火の test は、訓練にない言い回し、否定、訂正、曖昧な要求を含める。
- 社内 RAG は ACL をラベルではなく実行時にも強制する。モデル出力だけで閲覧可否を決めない。
- 人物の声を扱うため、録音原本、派生 token、チェックポイントへのアクセスを分離し、削除要求に対応できるデータ台帳を持つ。

## 8. 段階的な実装計画

| 段階 | 実施内容 | 成功条件 |
| --- | --- | --- |
| 0. 前提確認 | LLM-jp-Moshi-v1（Apache-2.0）の重み、対象音声の許諾、GPU、推論サーバーを確認 | 商用可否とデータ利用範囲が文書化される |
| 1. 音声ベースライン | LLM-jp-Moshi-v1 を動かし、対象話者データで小規模追加学習 | 音声・日本語会話・割り込みのベースラインを保存 |
| 2. 学習なし tool PoC | 外部分類器／バックエンド LLM がツール選択し、MoshiRAG 型で結果を注入 | UX、遅延、検索品質、キャンセル設計を検証 |
| 3. 制御 token PoC | 1 種類の `<TOOL_SEARCH>` を追加し、オフラインの検索結果で教師学習 | token の precision/recall と自然な待機応答を確認 |
| 4. RAG/LLM 拡張 | `<TOOL_RAG>` と `<TOOL_LLM>`、ACL、失敗処理、出典処理を追加 | 実会話で誤発火・情報漏えい・stale 注入が許容範囲 |
| 5. 並列知能拡張 | 必要性が確認された場合のみ KAME 型 oracle stream を研究 | 長い backend 処理中にも会話品質を維持 |

段階 2 は、モデル内発火を最終目標にしつつ、学習に入る前にプロダクト体験とバックエンドの失敗モードを検証するために置く。外部分類器方式を最終形と取り違えないこと。

## 9. 評価指標

### ツール利用

- intent ごとの token precision、recall、F1
- 不要な発火率、同一要求の重複発火率
- 発火から待機発話までの時間、ツール結果注入から回答開始までの時間
- 割り込み後のキャンセル率と stale 結果の誤注入率
- RAG の根拠適合率、アクセス拒否の正確性、出典提示率

### 音声・対話

- 対象話者との話者類似度（自動指標と人手評価）
- 音質、聞き取りやすさ、感情・話速・相槌の自然さ
- 同時発話・割り込み・ターン終了の人手評価
- ツール token の直前／直後で音声が崩れないか
- p50/p95 の end-to-end 遅延と GPU 使用量

ベースライン、段階 2、段階 3 以降を同じ評価セットで比較し、声の適応が tool use や会話品質を悪化させていないかを必ず測る。

## 10. リスクと意思決定ポイント

| リスク | 影響 | 対応 |
| --- | --- | --- |
| 誤って J-Moshi の非商用重みを使う | 商用化不能 | LLM-jp-Moshi-v1 を明示して固定し、取得元・hash・ライセンスを記録する |
| 制御 token の誤発火 | 不要な検索、情報漏えい、費用増 | `NONE` 負例、サーバー側の tool policy、rate limit、確認フロー |
| 検索結果の遅延 | 不自然な沈黙 | 短い待機応答、非同期処理、タイムアウト応答 |
| 古い結果の注入 | 訂正後に誤答 | conversation revision、キャンセル、stale discard |
| 声データの権利 | 人格権・契約・削除要求 | 明示許諾、データ台帳、アクセス制御、削除手順 |
| 追加学習による能力劣化 | 日本語会話や tool 判定が悪化 | 段階学習、保持データ、回帰評価、checkpoint 比較 |
| GPU/運用費 | 検証が進まない | まず 1 token・短い会話・小規模データで計測 |

## 11. 参考資料

### LLM-jp-Moshi / J-Moshi / Moshi

- [本リポジトリ README（英語）](../README.md)
- [本リポジトリ README（日本語）](../README-ja.md)
- [LLM-jp-Moshi-v1 モデルカード（Apache-2.0）](https://huggingface.co/llm-jp/llm-jp-moshi-v1)
- [LLM-jp-Moshi 公式リポジトリ](https://github.com/llm-jp/llm-jp-moshi)
- [NII: 商用利用可能な LLM-jp-Moshi-v1 の公開](https://www.nii.ac.jp/news/release/2026/0225.html)
- [J-Moshi モデルカード](https://huggingface.co/nu-dialogue/j-moshi-ext)
- [Kyutai Moshi](https://github.com/kyutai-labs/moshi)
- [MoshiRAG](https://github.com/kyutai-labs/moshi-rag)
- [MoshiRAG 論文](https://arxiv.org/abs/2604.12928)
- [KAME: Knowledge-Augmented Multi-stream End-to-end spoken dialogue](https://arxiv.org/pdf/2510.02327)
- [KAME 推論実装](https://github.com/SakanaAI/kame)
- [KAME 追加学習実装](https://github.com/SakanaAI/kame_finetune)

### GPT Live の体験設計

- [Introducing GPT Live](https://openai.com/ja-JP/index/introducing-gpt-live/)

### VoiceChat の比較調査

- [NemotronLabs VoiceChat README](https://github.com/NVIDIA-NeMo/Speech/tree/nemotron-labs-voicechat)
- [VoiceChat モデルカード](https://huggingface.co/nvidia/NVIDIA-NemotronLabs-VoiceChat-11B)
- [EARTTS 学習スクリプト](https://github.com/NVIDIA-NeMo/Speech/blob/nemotron-labs-voicechat/examples/speechlm2/duplex_eartts_train.py)
- [Voice lock 実装](https://github.com/NVIDIA-NeMo/Speech/blob/nemotron-labs-voicechat/examples/speechlm2/nemotron-labs-voicechat_infer_voice_lock.py)
- [STT/TTS/RNNT checkpoint 結合スクリプト](https://github.com/NVIDIA-NeMo/Speech/blob/nemotron-labs-voicechat/examples/speechlm2/combine_ckpt_conda.sh)
- [VoiceChat 運用前提](https://github.com/NVIDIA-NeMo/Speech/blob/nemotron-labs-voicechat/docs/source/voicechat_realtime_instructions/prerequisites.md)
- [OpenMDW 1.1 ライセンス](https://github.com/NVIDIA-NeMo/Speech/blob/nemotron-labs-voicechat/LICENSE_OpenMDW-1.1)

## 12. 次の実装タスク（未着手）

1. LLM-jp-Moshi-v1 の checkpoint と本リポジトリの互換性を確認し、tokenizer と既存 special token を調べ、予約 token 利用か語彙拡張かを決める。
2. MoshiRAG をローカルに読み、LLM-jp-Moshi-v1 の推論サーバーへ最小の `<TOOL_SEARCH>` event loop を設計する。
3. 「発火」「待機応答」「結果注入後の回答」を含む小規模な教師データ仕様を決める。
4. tool token の追加で変更するモデル初期化・データ前処理・推論デコーダ・テストの一覧を作る。
5. 使用する LLM-jp-Moshi-v1 のモデル revision、Apache-2.0 ライセンス、対象話者と追加学習データの利用許諾を記録する。

## 13. MoshiRAG の詳細調査

### 13.1 アプローチ: 選択的・非同期 retrieval

MoshiRAG は、検索や RAG が必要なときだけモデル自身が retrieval trigger を出す、モジュール型の full-duplex 音声エージェントである。会話を止めてから検索するのではなく、検索を開始したあとも Moshi が音声を聞き、短い相槌・待機発話・粗い回答（`pre-RAG content`）を生成し続ける。

```text
ユーザー音声
  │
  ├──► Moshi の入力音声 stream
  │        ├── 通常の音声・内部 text を生成
  │        └── 専用 retrieval token を生成
  │
  └──► streaming ASR ──► 会話文脈
                                 │
                    （非同期・必要時のみ）
                                 ▼
                 検索 / RAG / retrieval LLM
                                 │ reference text
                                 ▼
                 reference encoder / projection
                                 │ streaming condition
                                 ▼
                  後続の Moshi 音声生成を根拠付け
```

論文上の狙いは、ユーザー発話終了からモデルが重要語を言い始めるまでの **keyword delay** を retrieval の待ち時間として使うことである。従って、`<RET>` の直後に「少し確認しますね」などを自然に生成できれば、retrieval の待ち時間を会話上ほぼ隠せる。[MoshiRAG 論文](https://arxiv.org/abs/2604.12928)

### 13.2 実行時の実装

公開 Moshika-RAG checkpoint の `config.json` では `rag_token_id: 4` が専用 ID として設定されている。`Channel` は、Moshi がこの ID を出した時点で以下を行う。

1. UI には `[RET]` event を送り、streaming ASR を flush する。
2. per-channel の `RAGManager` を起動する。発火直後に古い未完了タスクがあればキャンセルする。
3. `TurnManager` が保持する、Moshi の内部 text と ASR transcript を組み合わせた会話文脈を retrieval back end へ渡す。
4. 得られた reference text を UI に出し、別プロセスの `server_conditioner` へ HTTP で渡して埋め込み化する。
5. 返った tensor を対象セッションの `streaming_sum` condition として以後の生成ステップへ適用する。

retrieval 自体は `asyncio` の background task であり、標準 server 設定の reference 生成 timeout は 1.5 秒である。失敗・timeout 時は空の reference を返して会話ループを止めない。retrieval LLM は OpenAI 互換 endpoint を使えるほか、複数 profile と fallback を設定できる。コード上の back end は text-in/text-out であるため、Web 検索や社内 RAG はここへ実装できる。[発火・結果注入コード](https://github.com/kyutai-labs/moshi-rag/blob/main/moshi/moshi/inference_utils/channel.py) [非同期タスク管理](https://github.com/kyutai-labs/moshi-rag/blob/main/moshi/moshi/inference_utils/rag_manager.py) [reference encoder](https://github.com/kyutai-labs/moshi-rag/blob/main/moshi/moshi/server_conditioner.py)

reference は単なるプロンプト文字列ではない。MoshiRAG は reference text encoder と bridge projection を持ち、得た表現をモデルの `streaming_sum` conditioner として時系列に加算する。このため retrieval が途中で届いても、それ以降の音声トークンを根拠付きにできる。

### 13.3 学習方法と公開範囲

論文では、LLM で会話スクリプトと reference document を生成し、二話者の conversational TTS で音声化した synthetic conversation を作る。RAG 対象 turn では、forced alignment を用いて回答の lead 部分にある text token の直前を retrieval token に置き、モデルが重要情報を話す前に発火するよう教師学習する。reference conditioner には、到着時刻のずれを想定した時間 sampling、reference dropout もある。

公開リポジトリは PyTorch / Rust の**推論**実装、reference encoder、UI、Moshika-RAG checkpoint を提供している。一方、汎用的な追加学習・データ生成の一式は公開リポジトリの中心には含まれない。したがって LLM-jp-Moshi-v1 に移植する際は、trigger token の教師データ、reference-with-time conditioner、model の再学習を本ブランチで設計する必要がある。公開 Moshika-RAG weight は synthetic female voice 向けであり、LLM-jp-Moshi-v1 の日本語話者モデルをそのまま置換するものではない。[MoshiRAG repository](https://github.com/kyutai-labs/moshi-rag)

### 13.4 本件への移植

MoshiRAG の `<RET>` を、以下の明示的な tool token へ一般化する。

| token | 実行 | MoshiRAG との対応 |
| --- | --- | --- |
| `<TOOL_SEARCH>` | 時事・公開情報の Web search | retrieval back end を検索実装にする |
| `<TOOL_RAG>` | ACL 付き社内文書 retrieval | retrieval back end を RAG 実装にする |
| `<TOOL_LLM>` | 長い推論・コード・構造化処理 | retrieval back end を backend LLM にする |

LLM-jp-Moshi-v1 では、Moshika-RAG の ID 4 をハードコードしてはならない。SentencePiece tokenizer と special token の予約状況を検査し、専用語彙の追加または安全な既存 ID の確保を行う。結果は必ず可聴・UI 用の出力から除外し、サーバー側で action allowlist、ACL、rate limit、キャンセル、stale-result discard を強制する。

## 14. KAME の詳細調査

### 14.1 アプローチ: tandem architecture と oracle stream

KAME は、低遅延 S2S front end と高知識の backend LLM を同時に動かす tandem architecture である。Moshi の既存 three-stream（入力音声、出力音声、内部 text）に、backend LLM の逐次回答候補を表す **第4の oracle stream** を追加する。

```text
ユーザー音声 ─┬─► Moshi / KAME front end ─► 即時の相槌・音声応答
              │       ▲
              │       │ oracle token stream
              ▼       │
       streaming ASR ─┴─► backend LLM（回答候補を逐次生成）
```

oracle は front end が予測する出力ではなく、外部 LLM が書き込む条件入力である。モデルコードでは `oracle_emb` を追加し、各時刻の oracle token embedding を既存の text embedding へ加算して Transformer へ入力する。これにより先に出た曖昧な回答を、後から到着したより正確な LLM 候補で修正・具体化できる。[KAME model code](https://github.com/SakanaAI/kame/blob/main/src/kame/models/lm.py)

### 14.2 実行時の実装

現行の `server_oracle.py` は、Google Cloud Speech-to-Text の英語 ASR（`en-US`）の interim transcript を受けるたび、OpenAI Chat Completions の streaming request を起動する。標準では最短 0.5 秒間隔で再起動し、最大 7 本の generation を重ね、より新しい generation が最初の token を返すとそれを採用する。

採用された LLM の text chunk は queue を経由し、音声処理ループだけが `lm_gen.update_oracle_tokens_streaming()` を呼ぶ。新しい generation へ切り替わる時は oracle を reset してから token を追加する。この single-writer 設計により、LLM の非同期到着と audio generation の race を避けている。現公開コードは `gpt-4.1`、Google ASR、単一 WebSocket session を前提とするため、日本語・複数セッション・自前 LLM で使うには置換が必要である。[KAME oracle server](https://github.com/SakanaAI/kame/blob/main/src/kame/server_oracle.py)

ここには MoshiRAG のような「モデルが tool token を出したら検索する」という選択的発火はない。backend LLM を ASR partial に対して先回り実行する方式であり、会話文脈の必要性とは独立に LLM 呼び出しが増える。Web search / RAG を組み込みたい場合は、backend LLM に tool calling を実装するか、別の orchestrator を追加する必要がある。

### 14.3 学習方法

自然な対話データには、発話途中で得られる不完全な backend 回答は存在しない。そこで KAME は standard two-party dialogue を次の oracle sequence へ変換する。

1. ユーザー発話の途中までで、simulator LLM がもっともらしい暫定回答を生成する。
2. 聞き取れた語の割合が増えるにつれ、ground-truth 応答の hint を段階的に強く与え、候補を具体化する。
3. ユーザー発話完了時には ground-truth 応答へ収束させる。
4. TTS と word alignment でこの oracle event を音声 token の時刻へ配置する。

追加学習コードは oracle token を `oracle_emb`（または text embedding と共有する tie mode）で入力するが、oracle 自体の予測損失は持たない。通常の text/audio cross-entropy を最小化し、「oracle を受けて自然な音声応答を生成する」ことを学習する。実環境の遅延・欠落に耐えるため、oracle の右／左 shift、time jitter、event skip、semantic/text embedding dropout も扱う。[KAME 論文](https://arxiv.org/abs/2510.02327) [KAME finetuning code](https://github.com/SakanaAI/kame_finetune/blob/main/finetune.py)

論文の英語・音声合成 MT-Bench 評価では、Moshi 単体の平均 score 2.05 に対し、GPT-4.1 を用いた KAME は 6.43、カスケード baseline は 7.70 だった。KAME は低遅延を維持する一方、完全なユーザー発話を待たずに話し始めることによる早すぎる回答が、カスケードとの差として残る。

## 15. 比較と採用順序

| 観点 | MoshiRAG | KAME |
| --- | --- | --- |
| 外部処理の開始 | モデルが trigger token を出した時だけ | ASR partial ごとに backend LLM を先行実行 |
| 外部から戻す内容 | 検索結果・RAG document・要約などの reference | backend LLM の逐次的な回答候補 |
| モデルへの戻し方 | reference encoder の `streaming_sum` condition | 第4 stream の `oracle_emb` を text embedding に加算 |
| 学習改変 | trigger token と reference conditioner を学習 | `oracle_emb` を追加し、oracle event を含む時系列データで学習 |
| 実行コスト | 必要時のみ tool を実行 | partial ごとに LLM call が発生し得る |
| Web/RAG への適性 | 高い。back end を差し替えやすい | 直接は低い。LLM 側に tool orchestration が必要 |
| LLM の深い逐次統合 | reference を受けた後続発話が中心 | 高い。途中から候補を更新し続けられる |

本件の実装順は、まず **LLM-jp-Moshi-v1 + MoshiRAG 型**とする。

1. 学習なしの orchestrator で Web search / RAG / backend LLM の非同期 UX、キャンセル、出典、遅延を検証する。
2. `LLM-jp-Moshi-v1` と現在の追加学習実装の checkpoint / tokenizer / model-shape 互換性を確認する。
3. 1 種類の `<TOOL_SEARCH>` と reference result injection を追加学習する。
4. `<TOOL_RAG>`、`<TOOL_LLM>` と ACL、tool policy、評価を追加する。
5. 「外部 LLM の逐次回答を会話の途中で何度も取り込む」価値が実証された場合にのみ、KAME の oracle stream を日本語モデルへ移植する。

この順序なら、tool use の誤発火・検索遅延・話者音声の品質低下を段階的に評価でき、最初から KAME のモデル構造変更と synthetic oracle data generation を抱え込まずに済む。
