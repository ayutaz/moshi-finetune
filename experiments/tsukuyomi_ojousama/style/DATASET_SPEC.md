# お嬢様口調100対話 再生成仕様

更新日: 2026-08-18

過去実験の`ojousama_mild_100` script本体が回収できない場合に限り、本仕様で100対話を新規作成する。`OjousamaTalkScriptDataset`の201件は品質参照専用であり、completionの転載・言い換えや固定評価promptの再利用はしない。

## 規模と分割

- 10 topics × 10 dialogues = 100 dialogues。
- topics: 天気、食事、仕事、家族、旅行、趣味、健康、買い物、技術、日常雑談。
- 1 dialogueはA/B各4発話を目安に8 turns、合計約800 turns。
- Aはつくよみちゃん系のお嬢様話者、Bは中立的で丁寧な対話相手。
- 対話単位で80/10/10に分け、同じscenario、固有名詞、定型導入の近似文をsplit間に置かない。

各dialogueには一意な`dialogue_id`、`topic`、`scenario`、`variant`、`generation_model`、`prompt_sha256`、`created_at`と、各turnの`speaker`、`text`を持たせる。採用前にJSON schema、話者交替、空発話、完全・近似重複、固定評価との重複を検査する。

## 口調variant

| variant | A発話の目標 | 禁止事項 |
| --- | --- | --- |
| strict | `わたくし`、`ですわ`、`ますわ`、`かしら`、`ですの`等を90%以上の発話に含める | 普通体、同一語尾3連続、意味のない語尾付加 |
| mild | お嬢様markerを60〜75%の発話に自然に含める | 同一語尾3連続、全発話への機械的な`ですわ`付加 |

両variantとも、質問への回答内容、対話の一貫性、敬語の文法を口調markerより優先する。Aが毎回同じ挨拶で始める、Bが執事という単一関係だけに偏る、会話を締める定型句を反復する、といったshortcutを禁止する。

## 生成手順

1. Codexでtopicごとに重複しないscenario表を先に作る。
2. scenarioとvariantを固定したpromptから、1対話ずつJSONとして生成する。
3. 自動検証で件数、schema、marker率、語尾連続、重複、固定評価漏洩を確認する。
4. dev/test全件とtrainの20%を人手で読み、意味、自然さ、敬語、話者一貫性を採点する。
5. 不合格対話だけを新しいseedで再生成し、棄却理由もreportに残す。
6. 採用scriptを同一分割のまま、A=採用Tsukuyomi TTS、B=固定中立TTSで24 kHz stereoへ音声化する。
7. forced alignment後、音声・text・speaker channel・時刻の一致を検証してからtokenizeする。

## 音声化の固定条件

- A/BのTTS checkpoint、speaker ID、sampling設定、seed、音量正規化条件をmanifest化する。
- COEIROINK出力は使用しない。
- A/Bの音声が重なる箇所、無音、応答間隔も生成manifestに記録する。
- 原音、未公開TTS checkpoint、生成音声は公開条件の再監査までは非公開とする。

## 採用Gate

- 100 dialogues、約800 turnsがschema検証を通る。
- 完全重複、split間の近似重複、固定評価との漏洩が各0件。
- strictはA発話marker率90%以上、mildは60〜75%。
- 同一語尾3連続、意味破綻、話者逆転が各0件。
- 人手監査対象のrelevanceとnaturalnessが各5段階平均4.0以上。

このGateを満たすまで、Moshiのstyle fine-tuningへ投入しない。
