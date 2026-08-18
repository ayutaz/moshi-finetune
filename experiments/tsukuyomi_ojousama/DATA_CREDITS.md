# データクレジットと非公開対象

## つくよみちゃんコーパス

本実験の音声学習には、フリー素材キャラクター「つくよみちゃん」（© Rei Yumesaki）が無料公開している音声データを使用しています。

- つくよみちゃんコーパス（CV.夢前黎）
- https://tyc.rei-yumesaki.net/material/corpus/

原音ZIP、展開したWAV、および原音から作成した素材用音声はGitやデータセットとして再配布しない。第三者が声質を利用できるcheckpointを公開する場合は、公開前に公式規約の「声質を使用した音声合成ソフト等を公開する」の条件を再監査し、規定のクレジット・禁止事項・規約継承をmodel cardへ記載する。

## お嬢様口調テキスト

本実験でCodexが新規作成する評価promptと会話scriptは、つくよみちゃんコーパスの原音・台本とは分離して管理する。`OjousamaTalkScriptDataset`を参照または派生元として使用する場合は、MIT License、取得commit、checksumを台帳へ記録する。

- 参照データ: `matsuvr/OjousamaTalkScriptDataset`
- 固定commit: `589f3b52324cc12ad3fb0b2ebe1520bbffce4087`
- License: MIT
- Copyright (c) 2023 matsu_vr

このデータは`reference-only`とし、固定評価promptや今後生成する学習会話と混ぜない。元ファイルの202行のうち重複prompt 1件を除いた201件を品質参照用に固定している。

## 固定評価テキスト

`eval/`の未見TTS 30文、お嬢様口調50 pair、一般対話30件は、この実験用にCodexで新規作成した。つくよみちゃんコーパスの台本および`OjousamaTalkScriptDataset`のcompletionを転載・派生していない。Apache-2.0の本リポジトリ内成果物として管理し、学習データからは常に除外する。声質参照indexに対応する原音WAVは非公開のままとする。

## 非公開対象

- つくよみちゃんコーパスの原音ZIPとWAV
- API key、SSH private key、Vast.aiアカウント情報
- 公開条件を満たすと確認する前の派生checkpointと生成音声
