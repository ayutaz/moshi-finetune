# つくよみちゃん／お嬢様実験の終了記録

終了日: 2026-10-04。利用者の指示により、追加学習を行わず成果物を保存して終了する。
実験の目的を達成したという判定ではない。

## 最終結果

| 工程 | 終了時点の結果 |
| --- | --- |
| M0 / M1 | baseline回収、データ・権利・固定評価セットの整備を完了 |
| M2 | TTSを完了。採用は speaker inversion（12,288パラメータ、base凍結） |
| M3 | Voice controlは不合格。原因の旧診断は撤回済み |
| M3-R | 計器・データ・記録の修復とbase loss測定を完了。run1は学習開始前に停止し、checkpointなし。未完了のまま中止 |
| M4〜M6 | 未実施。release candidateなし |

数値と判定の根拠は[マイルストーン](./j-moshi-tsukuyomi-ojousama-milestones.md)、
[M3検証記録](./j-moshi-tsukuyomi-ojousama-m3-verification.md)、
[M3-R現在地](./j-moshi-tsukuyomi-ojousama-m3r-status.md)に残す。
GPU費用は台帳上 **US$107.301**。終了作業ではGPUを借りていない。

## 保存先

- コード・実験計画・報告・registry・manifest・終了記録:
  [GitHubの実験ブランチ](https://github.com/ayutaz/moshi-finetune/tree/experiment-j-moshi-character-voice-overfit)
- 未送信だった調査文書:
  [GitHubの調査ブランチ](https://github.com/ayutaz/moshi-finetune/tree/research/j-moshi-tool-token-finetuning)
- モデル・生成評価出力・ログ・再構築用メタデータ:
  [HF公開アーカイブ（手動承認ゲート）](https://huggingface.co/ayousanz/moshi-tsukuyomi-ojousama-archive-2026-10-04)

HFの保存結果と検証値は
[`reports/project-closeout-2026-10-04.json`](../../experiments/tsukuyomi_ojousama/reports/project-closeout-2026-10-04.json)
を正本とする。**保存payloadは1,762ファイル / 62,327,508,773 bytes。HF側のサイズ・チェックサムとの照合は欠損0・不一致0。**
HFの`closeout/`にも終了記録と保存結果の写しを置く。GitHub / HFとも`project-closeout-2026-10-04` tagで終了時点を固定する。
HFは品質不合格の研究成果を公開・手動承認ゲートで保存するアーカイブであり、品質合格モデルのリリースではない。
GitHubではLFSを使用せず、モデル・音声の大容量payloadはHFだけに保存する。
初回は非公開ストレージ上限でcommitが失敗したため、利用者がpublic / manual gateへ変更し、
アップロード継続を明示した。詳細は上記JSONのpublication reviewと検証結果に残す。

保存対象は、M2の話者埋め込み13本とconfig、M3のV-real / V-ttsそれぞれepoch 3 / 5の
checkpoint計4本とconfig、回収済み生成token・評価音声、学習・診断ログ、再構築用メタデータ、
Gitのsource snapshot。ファイルごとの元パス・サイズ・SHA-256をHFの`archive-manifest.json`に記録し、
`SHA256SUMS`を併置する。

## 復元

認証済みの所有者、または手動ゲートで承認されたHFアカウントで実行する。

```bash
hf download ayousanz/moshi-tsukuyomi-ojousama-archive-2026-10-04 --local-dir restored
cd restored
shasum -a 256 -c SHA256SUMS
```

`source-*.tar.gz`を展開し、保存した`data/`を同じ相対パスへ戻す。
base modelは各run manifestの固定revisionから再取得する。
これは推論weightの保存であり、optimizer stateを持つ学習再開用checkpointではない。

## 保存対象外と欠損

原音ZIP/WAV、原音を含むV-realの音声・token・parquet（旧版を含む）、自然音声参照、
原音を含む評価prompt、M2のcorpus latent、corpusのroom-tone断片はHFへ送らずローカルに残す。
独立して上流corpusを入手し、コミット済みの手順とmanifestから再構築する。
生成評価出力と声質由来モデルは、利用者の明示指示により公開・手動承認ゲートで保管する。
生成音声・tokenは報告された実験の聴取・評価用で、素材の二次利用・再配布や学習用datasetとしての許可は与えない。
model cardに上流クレジット・4禁止用途・規約継承・混合声質の説明を記載した。
原音・原音を含む素材の扱いは[公式利用規約](https://tyc.rei-yumesaki.net/material/corpus/)と
[DATA_CREDITS.md](../../experiments/tsukuyomi_ojousama/DATA_CREDITS.md)に従う。
認証情報、cache、仮想環境、再取得可能なbase weightも保存対象外。
ローカルファイルは削除していない。

M3のepoch 1 / 2 / 4のweightは当時exportされず、instanceも破棄済み。
V-tts epoch 2も復元不能。評価数値と生成物の存在をweightの保存と混同しない。
M3-R run1はcheckpointを作っておらず、bootstrap時の依存version記録も欠損している。
NCCL P2P原因説は未検証のまま。

## 計算資源と検証

台帳では実験用instanceはすべて破棄済みで、日次課金US$0と記録されている。
2026-10-04のVast.aiライブ照会はHTTP 401（既存キー無効）で失敗したため、現在の状態を
再確認できたとは扱わない。別プロジェクトのinstanceは操作していない。

終了時のテスト: **974 passed / 8 skipped / 22 subtests passed**。lint / formatも合格。
`finetune.py`の既存のlist内包表記2か所を現行formatterに合わせた（動作変更なし）。
skipは既存の条件付きテストによるもの。成果物のHF側検証結果は上記JSONに記録する。
過去の計画、費用、失敗記録は履歴として残し、完了条件を緩めていない。
