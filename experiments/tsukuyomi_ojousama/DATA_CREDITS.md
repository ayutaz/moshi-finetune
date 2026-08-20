# データクレジットと非公開対象

## つくよみちゃんコーパス

本実験の音声学習には、フリー素材キャラクター「つくよみちゃん」（© Rei Yumesaki）が無料公開している音声データを使用しています。

- つくよみちゃんコーパス（CV.夢前黎）
- https://tyc.rei-yumesaki.net/material/corpus/

原音ZIP、展開したWAV、および原音から作成した素材用音声はGitやデータセットとして再配布しない。第三者が声質を利用できるcheckpointを公開する場合は、公開前に公式規約の「声質を使用した音声合成ソフト等を公開する」の条件を再監査し、規定のクレジット・禁止事項・規約継承をmodel cardへ記載する。

### 追加候補: 夢前黎の音声データの寄せ集め（未取得）

- 取得元: https://tyc.rei-yumesaki.net/material/corpus/yoseatsume/
- 版: 公開2024-06-18 / 更新2024-06-20（2026-08-20時点で確認）
- 規模: 約1,500台詞。つくよみちゃんコーパスVol.1の100文、JSUT basic5000の夢前黎担当600文、その他朗読等を含む
- 入手: **申請制**。上流のメールフォームから申請し、返送されるダウンロードURLを受け取る
- 利用条件: 音声技術（合成・声質変換・認識等）の研究開発のみ。声質を表に出さない場合はクレジット不要、出す場合は「つくよみちゃんから提供された他の音声データ」等と記載する
- 禁止事項: **第三者への公開・譲渡禁止**

現在は未取得で、manifestにも1件も含まれていない。台帳は`registry/tsukuyomi-yoseatsume-candidate.json`にあり、`used_in_experiment: false`として管理する。

取得する場合は次の2点に注意する。第一に、この寄せ集めはつくよみちゃんコーパスVol.1の100文を**含む上位集合**なので、`tsukuyomi-corpus-v1`と重複排除してから分割しないと、固定held-outがtrainへ再流入する。第二に、JSUT basic5000部分は別の上流条件を持つため、使用前に別途台帳化する。取得後の音声とその派生物は再配布禁止のため、下記「非公開対象」に加える。

## お嬢様口調テキスト

本実験でCodexが新規作成する評価promptと会話scriptは、つくよみちゃんコーパスの原音・台本とは分離して管理する。`OjousamaTalkScriptDataset`を参照または派生元として使用する場合は、MIT License、取得commit、checksumを台帳へ記録する。

- 参照データ: `matsuvr/OjousamaTalkScriptDataset`
- 取得元: https://github.com/matsuvr/OjousamaTalkScriptDataset
- 固定commit: `589f3b52324cc12ad3fb0b2ebe1520bbffce4087`
- License: MIT
- Copyright (c) 2023 matsu_vr
- 許諾条項本文: [`reference/LICENSE.OjousamaTalkScriptDataset`](./reference/LICENSE.OjousamaTalkScriptDataset)（上流LICENSEの逐語コピー、SHA-256 `fcd8fbf3ea9a8a08b3c5ffd6ede9d6d129b006eacf190c54202673f2906e207f`）

このデータは`reference-only`とし、固定評価promptや今後生成する学習会話と混ぜない。元ファイルの202行のうち重複prompt 1件を除いた201件を品質参照用に固定している。

派生物である`reference/ojousama-talk-script-201.jsonl`は本リポジトリに含めて公開しているため、MITの「上記著作権表示および本許諾表示を、ソフトウェアのすべての複製または重要な部分に記載する」条件が適用される。著作権表示だけでは足りないので、許諾条項本文を上記のファイルとして同梱する。この派生物を別の場所へ再配布する場合も、同じ2つを併せて配布すること。

## 固定評価テキスト

`eval/`の未見TTS 30文、お嬢様口調50 pair、一般対話30件は、この実験用にCodexで新規作成した。つくよみちゃんコーパスの台本および`OjousamaTalkScriptDataset`のcompletionを転載・派生していない。Apache-2.0の本リポジトリ内成果物として管理し、学習データからは常に除外する。声質参照indexに対応する原音WAVは非公開のままとする。

## 非公開対象

- つくよみちゃんコーパスの原音ZIPとWAV
- API key、SSH private key、Vast.aiアカウント情報
- 公開条件を満たすと確認する前の派生checkpointと生成音声
