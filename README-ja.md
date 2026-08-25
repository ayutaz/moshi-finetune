# Moshi/J-Moshi Finetuning

[**English README**](README.md) | [**日本語 README**](README-ja.md)

Kyutai が提案した Full-duplex 音声対話モデル Moshi をファインチューニングするための非公式リポジトリです。RQ-Transformer を所望の音声対話データで学習することができます．また，Moshi をベースに日本語データで学習された J-Moshi（詳しくは，[Finetuned Model](#finetuned-model)を参照）のファインチューニングも可能です．本リポジトリの学習スクリプトは，以下で公開されている，Moshiの公式の[テクニカルレポート](https://arxiv.org/abs/2410.00037)および [Pytorch モデル](https://github.com/kyutai-labs/moshi)をベースに再現実装されました：

なお，[公式のファインチューニングコード](https://github.com/kyutai-labs/moshi-finetune) と比較した際の，本リポジトリの特徴は以下の通りです：
- 分散学習の実装においては，公式コードベースで用いられている [Fully Sharded Data Parallel (FSDP)](https://pytorch.org/blog/introducing-pytorch-fully-sharded-data-parallel-api/) ではなく，[🤗 Accelerate](https://github.com/huggingface/accelerate) と [DeepSpeed](https://github.com/deepspeedai/DeepSpeed) を用いています．これらのライブラリに慣れている方にはおすすめです．
- 我々のコードベースでは，公式のテクニカルレポートと同様に，Moshi 自身のストリームと並列してユーザの音声ストリームの生成も学習します．これによって，prompted dialogue continuation などの自動評価が容易になります．

## Finetuned Model
このリポジトリを用いた Moshi のファインチューニングによって，日本語版モデルである J-Moshi が構築されました．J-Moshi は，69k 時間の日本語音声対話を含む [J-CHAT コーパス](https://huggingface.co/datasets/sarulab-speech/J-CHAT)），および，数百時間の日本語音声対話コーパスで学習されたモデルです．J-CHAT の学習では，128枚のNVIDIA V100 32GB GPU を用いて，およそ36時間を要しました．J-Moshi の詳細は以下のリンクを参照してください：
- [ウェブサイト](https://nu-dialogue.github.io/j-moshi)
- [学習済みモデル](https://huggingface.co/nu-dialogue/j-moshi-ext)
- [学習ログ](https://api.wandb.ai/links/ohashi56225/ty0dw2il)


## Environment Setup
Python 3.12+ required

### Dependencies
#### Option 1. Install with uv (recommended)
```bash
uv sync --python 3.12
```
uv のインストール方法や使用方法は，[公式ドキュメント](https://docs.astral.sh/uv/getting-started/)を参照してください．

#### Option 2. Install with pip
```bash
pip install -r requirements.txt
```

### Experiment Tracking
学習ログのモニタリングには [Weights & Biases (W&B)](https://wandb.ai/site) を使用しています．W&B を使用する場合は，以下のコマンドを用いて，W&Dのアカウントにログインしてください：
```bash
wandb login
```

## Usage
ここでは，[kyutai/moshiko-pytorch-bf16](https://huggingface.co/kyutai/moshiko-pytorch-bf16) をファインチューニングするための手順を説明します．なお，この手順では，uv を使用した例を示します．uv を使用しない場合は，`uv run` を `python` に置き換えるなど，自分の環境に合ったコマンドを使用してください．

### 1. Data Preparation
音声対話データ，および，その書き起こしテキストを離散トークンにトークナイズして，ファインチューニング用のデータセットを構築します．サンプルデータとして，[SpokenWOZ](https://arxiv.org/abs/2305.13040) から抽出したデータを使用します．詳細は [`data/spokenwoz_sample`](data/spokenwoz_sample) を参照してください．

#### 1.1. Audio Tokenization
音声対話データは，各対話における2話者（A and B）の音声がそれぞれのチャネルに分離されたwavファイルです．左チャネルに話者Aの音声，右チャネルに話者Bの音声が含まれている必要があります．1つのwavファイルに1つの対話データを含んでください．
具体的なデータの例は `data/spokenwoz_sample/audio/*.wav` を参照してください．

以下のスクリプトを使用して，ディレクトリ内のすべてのwavファイルを Mimi のエンコーダによって離散トークンに変換します．
```bash
uv run -m tools.tokenize_audio \
    --audio_dir data/spokenwoz_sample/audio \
    --output_dir data/spokenwoz_sample/tokenized_audio
```

これにより，`data/spokenwoz_sample/tokenized_audio/*.npz` が作成されます．各npzファイルには，AとBの音声トークンが保存されています：
```python
>>> import numpy as np
>>> npz = np.load("data/spokenwoz_sample/tokenized_audio/SNG0072.npz")
>>> npz["A"].shape[0]
8 # levels of residual vector quantization
>>> npz["A"].shape[1]
1271 # frames of audio token streams (12.5Hz)
```

#### 1.2. Text Tokenization
テキストデータ（json）は，タイムスタンプが付与された単語単位の書き起こしデータです．1つのjsonファイルに1つの対話データ，すなわち，2話者（AとB）の両方の書き起こしが含まれている必要があります．データのフォーマットは以下の通りです：
```json
[
  {"speaker": "A", "word": "hello", "start": 0.46, "end": 1.52},
  {"speaker": "B", "word": "hi", "start": 1.82, "end": 2.04},
  {"speaker": "B", "word": "customer", "start": 2.04, "end": 2.703},
  {"speaker": "B", "word": "service", "start": 2.703, "end": 3.145},
  {"speaker": "B", "word": "how", "start": 3.145, "end": 3.366},
  ...
]
```
この例のように，各単語には，話者（`speaker`），単語（`word`），開始時間（`start`），終了時間（`end`）が含まれています．`start`および`end`は，対応するwavファイルにおける秒数を示します．具体的なデータの例は `data/spokenwoz_sample/text/*.json` を参照してください．なお，データセットが単語単位のタイムスタンプを含んでいない場合は，[forced alignment](https://pytorch.org/audio/stable/tutorials/forced_alignment_tutorial.html) が実装された外部のライブラリ（例えば，[WhisperX](https://github.com/m-bain/whisperX)など）を使用して作成してください．これらライブラリの詳しい使い方については，それぞれのリポジトリを参照してください．

以下のスクリプトを使用して，ディレクトリ内のすべてのjsonファイルについて，そのテキストをトークンに変換します：
```bash
uv run -m tools.tokenize_text \
    --word_transcript data/spokenwoz_sample/text \
    --output_dir data/spokenwoz_sample/tokenized_text
```

これにより，`data/spokenwoz_sample/tokenized_text/*.npz` が作成されます．各npzファイルには，AとBのテキストトークンが保存されています：
```python
>>> import numpy as np
>>> npz = np.load("data/spokenwoz_sample/tokenized_audio/SNG0072.npz")
>>> npz["A"].shape
(1271,) # frames of text token stream (12.5Hz)
```

#### Tips: Use Other Text Tokenizers
Kyutaiが提供するテキストトークナイザ以外のトークナイザを使用したい場合は，`--text_tokenizer_repo` および `--text_tokenizer_name` を変更してください．なおトークナイザとしては SentencePieceモデルのみがサポートされています．例えば，J-Moshi では以下の設定を用いました：
```bash
uv run -m tools.tokenize_text \
    --word_transcript /path/to/japanese_corpus/text \
    --output_dir /path/to/japanese_corpus/tokenized_text \
    --text_tokenizer_repo rinna/japanese-gpt2-medium \
    --text_tokenizer_name spiece.model \
    --text_padding_id 3 \
    --end_of_text_padding_id 0 \
    --no_whitespace_before_word
```

> [!IMPORTANT]
> トークナイザの変更に伴い，padding トークンの ID（`--text_padding_id`），および end of padding トークンの ID （`--end_of_text_padding_id`）も変更する必要があるかもしれません．また，日本語や中国語など，単語間にスペースがない言語の場合 `--no_whitespace_before_word` フラグを使用してください．

#### 1.3. Concatenation of Audio and Text Tokens
音声トークン列とテキストトークン列を結合し，ファインチューニング用のデータセットを作成します．
```bash
uv run -m tools.prepare_dataset \
    --tokenized_text_dir data/spokenwoz_sample/tokenized_text \
    --tokenized_audio_dir data/spokenwoz_sample/tokenized_audio \
    --output_prefix processed_data/spokenwoz_sample/train
```
以上のコマンドにより，`processed_data/spokenwoz_sample/train-001-of-001.parquet` が作成されます．一つのparquetファイルには，最大100,000の対話が含まれます．データセットの構造は以下の通りです：
```python
>>> import numpy as np
>>> from datasets import load_dataset
>>> dataset = load_dataset("parquet", data_files="processed_data/spokenwoz_sample/train-001-of-001.parquet")["train"]
>>> dataset
Dataset({
    features: ['dialogue_id', 'A', 'B'],
    num_rows: 10
})
>>> dataset[0]["dialogue_id"]
'processed_data/spokenwoz_sample/train/SNG1640'
>>> np.array(dataset[0]["A"]).shape[0]
9 # 1 text stream + 8 audio streams
>>> np.array(dataset[0]["A"]).shape[1]
1036 # frames of text/audio token stream (12.5Hz)
```


### 2. Model Initialization
ここでは，Kyutaが公開しているモデルを，ファインチューニング用に初期化・編集します．この処理には主に以下の要素が含まれます：
- （テキストトークナイザを変更した場合）テキストトークンの埋め込みテーブルを初期化
    - `--init_text_embeddings` フラグを使用してください．
    - **現状では，語彙サイズの変更には対応していません．必ず語彙サイズが同じトークナイザを使用してください．**
- Depth Transformer に対して，ユーザの音声ストリームを出力するためのモジュールを追加
    - `--extend_modules_for_user_stream` フラグを使用してください
- DeepSpeed Zero 3 で学習するため，Transformer 内の一部のモジュールを修正（Monkey patch）

以下のコマンドを実行して，モデルを初期化します．
```bash
uv run -m tools.init_moshi_for_ft \
    --moshi_lm_repo kyutai/moshiko-pytorch-bf16 \
    --save_dir init_models/moshiko-both_streams-float32 \
    --model_dtype float32 \
    --extend_modules_for_user_stream
```
これにより，`init_models/moshiko-both_streams-float32` に初期化されたモデル（`model.safetensors`），およびその設定ファイル（`moshi_lm_kwargs.json`）が保存されます．
なお，bfloat16 に対応しないGPUを使用する場合は，`--model_dtype float32` を指定してください．


### 3. Training
1および2で作成したデータセットと初期化されたモデルを用いて，ファインチューニングを実行します．基本的には，🤗 Accelerate のランチャーを使用し，所望のプロセス数で実行してください：
```bash
uv run accelerate launch \
    --num_machines 1 \
    --num_processes 8 \
    --use_deepspeed \
    --deepspeed_config_file ds_configs/zero3-fp16-warmlr-act_ckpt.json \
    finetune.py \
        --output_dir "output/moshiko-finetuned" \
        --train_data_files "processed_data/spokenwoz_sample/train-*.parquet" \
        --model_dir "init_models/moshiko-both_streams-float32" \
        ...
```
現在 DeepSpeed 以外での学習はサポートしていないため，必ず `--use_deepspeed` および `--deepspeed_config_file` を指定してください．
バッチサイズや学習率，各トークンの損失の重み等は，`finetune.py` の引数として指定可能です．詳細は `uv run finetune.py --help` で確認してください．

> [!NOTE]
> 具体的な実行コマンドは，`examples/finetune_accelerate.sh` を参照してください．

学習中のチェックポイントは，DeepSpeed の状態ファイルとして保存されます．後段の推論で使用するためには，以下のスクリプトを用いて，その状態ファイルを safetensors 形式に変換してください：
```bash
uv run -m tools.zero_to_fp32 \
    output/moshiko-finetuned/step_10000 \
    output/moshiko-finetuned/step_10000_fp32 \
    --moshi_lm_kwargs_path init_models/moshiko-both_streams-float32/moshi_lm_kwargs.json
```
以上により，`output/moshiko-finetuned/step_10000_fp32` にモデルの重み `model.safetensors` およびその設定ファイル `moshi_lm_kwargs.json` が保存されます．

#### Tips: Multi-Node Training
複数ノードでの学習を行う場合は，すべてのノードで，`--machine_rank`, `--main_process_ip`, `--main_process_port`, そして `--deepspeed_multinode_launcher standard` の引数を追加した accelerate launch コマンドを実行してください．🤗 Accelerate によるマルチノード学習に関するさらなる詳細な使い方は，[公式ドキュメント](https://huggingface.co/docs/accelerate/basic_tutorials/launch#multi-node-training)を参照してください．

> [!NOTE]
> 複数ノード OpenMPI の mpirun で制御する場合は，`examples/finetune_mpi_accelerate.sh` を参照してください．もし accelerate launch を使用せず，全てのプロセスを mpirun で起動する場合は，`examples/finetune_mpi.sh` を参考にしてください．


## Inference
学習したモデルを用いて，音声対話の生成やリアルタイム対話を行うことができます．以下では，学習済みモデルを用いた例を示します．

### 1. Prompted Dialogue Continuation
[Data Preparation](#1-data-preparation) で作成したデータセットに含まれる数秒のプロンプトから，その続きを生成することができます． accelerate launch を使用して，以下のスクリプトを実行してください：
```bash
model_dir="output/moshiko-finetuned/step_10000_fp32"
uv run accelerate launch \
    --num_machines 1 \
    --num_processes 4 \
    generate.py \
        --output_dir "${model_dir}/continuation" \
        --model_dir "${model_dir}" \
        --eval_data_files "processed_data/spokenwoz_sample/test-*.parquet" \
        --prompt_length 125 \
        --generation_length 250 \
        --temperature 0.8 \
        --seed 20260825 \
        ...
```
`--prompt_length` および `--generation_length` を用いて，それぞれプロンプトの長さと生成テキストの長さを指定してください．データセットに含まれる各サンプルについて，その始まりから `--prompt_lenght` で指定された長さまでの部分がモデルに入力されます．単位は Mimi の1フレーム（80ms）です．その他，バッチサイズや温度パラメータ等の設定は，`generate.py` の引数として指定可能です．詳細は `uv run generate.py --help` で確認してください．

> [!NOTE]
> 具体的な実行コマンドは，`examples/generate_accelerate.sh` を参照してください．

#### Tips: Decode generated audio tokens to wav
生成されたトークンは，generate.py の引数の `--output_dir` で指定したディレクトリの下の `generated_tokens`内に，対話毎の npy ファイルとして保存されます．以下のスクリプトを使用して，これらの npy ファイルを wav ファイルに変換可能です：
```bash
uv run -m tools.decode_tokens \
    --tokens_dir "${model_dir}/continuation/generated_tokens" \
    --output_dir "${model_dir}/continuation/generated_wavs"
```
これにより，対話毎のwavファイルが保存されます．各wavファイルは，左チャネルにシステムの音声，右チャネルにユーザの音声をそれぞれ含んでいます．


### 2. Interactive Demo
[moshi ライブラリ](https://github.com/kyutai-labs/moshi)の web app（`moshi.server`）を使用して，学習したモデルとリアルタイムに対話することができます．以下の手順に従ってください．

#### 2.1 Cleaning the finetuned model
`moshi.server` でモデルを読み込むには，まず，[Model Initialization](#2-model-initialization) でファインチューニング用に編集したモジュールを，オリジナルのMoshiモデルの形式に戻す必要があります．具体的には，以下の処理が含まれます：
- Depth Transformer に追加した，ユーザの音声ストリームを出力するためのモジュールの削除
- DeepSpeed Zero 3 のために編集したモジュールの修正

以下のコマンドを実行して，モデルを元の形式に戻してください：
```bash
uv run -m tools.clean_moshi \
    --moshi_ft_dir output/moshiko-finetuned/step_10000_fp32 \
    --save_dir output/moshiko-finetuned/step_10000_cleaned \
    --model_dtype float32 \
    --remove_modules_for_user_stream
```
これにより，`output/moshiko-finetuned/step_10000_cleaned` に元の形式のモデル（`model.safetensors`）およびその設定ファイル（`moshi_lm_kwargs.json`）が保存されます．

#### 2.2 Running the server
以下のコマンドを実行して，サーバを起動してください：
```bash
uv run -m moshi.server \
    --moshi-weight output/moshiko-finetuned/step_10000_cleaned
```
デフォルトでは `http://localhost:8998` でサーバが起動します．ブラウザでこのURLにアクセスすることで，対話を開始することができます．詳しい使い方は，Kyutai公式のリポジトリ[kyutai-labs/moshi](https://github.com/kyutai-labs/moshi)を参照してください．


## License
このリポジトリは [Apache 2.0 License](LICENSE) の下で提供されています．なお，`data/spokenwoz_sample` に含まれる，SpokenWOZ のサンプルデータは [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) の下で提供されています．


## Citation
このリポジトリを使用した場合は，以下の文献を引用してください：
```bibtex
@article{ohashi2025towards,
    title={Towards a Japanese Full-duplex Spoken Dialogue System},
    author={Ohashi, Atsumoto and Iizuka, Shinya and Jiang, Jingjing and Higashinaka, Ryuichiro},
    journal={arXiv preprint arXiv:2506.02979},
    year={2025}
}
```
