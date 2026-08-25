# Moshi/J-Moshi Finetuning

[**English README**](README.md) | [**日本語 README**](README-ja.md)

This is an unofficial repository for finetuning Moshi, a full-duplex spoken dialogue model proposed by Kyutai. You can train the RQ-Transformer on your desired spoken dialogue data. It also supports finetuning J-Moshi, a Japanese model based on Moshi (see [Finetuned Model](#finetuned-model) for details). The training scripts in this repository were reimplemented based on the official [technical report](https://arxiv.org/abs/2410.00037) and [PyTorch model](https://github.com/kyutai-labs/moshi).

Compared to the [official finetuning code](https://github.com/kyutai-labs/moshi-finetune), this repository has the following features:
- For distributed training, we use [🤗 Accelerate](https://github.com/huggingface/accelerate) and [DeepSpeed](https://github.com/deepspeedai/DeepSpeed) instead of [Fully Sharded Data Parallel (FSDP)](https://pytorch.org/blog/introducing-pytorch-fully-sharded-data-parallel-api/) used in the official codebase. This is recommended for those familiar with these libraries.
- In our codebase, we train the generation of the user's speech stream in parallel with Moshi's own stream as in the official technical report. This facilitates automatic evaluation such as that by using prompted dialogue continuation.


## Finetuned Model
Through finetuning of Moshi using this repository, the Japanese model J-Moshi was built. J-Moshi is a model trained on the [J-CHAT corpus](https://huggingface.co/datasets/sarulab-speech/J-CHAT)), which contains 69k hours of Japanese spoken dialogues, as well as hundreds of hours of Japanese spoken dialogue corpora. Training on J-CHAT took approximately 36 hours using 128 NVIDIA V100 32GB GPUs. For more details on J-Moshi, please refer to the following links:
- [Website of J-Moshi](https://nu-dialogue.github.io/j-moshi)
- [Finetuned Model](https://huggingface.co/nu-dialogue/j-moshi-ext)
- [Training Logs](https://api.wandb.ai/links/ohashi56225/ty0dw2il)


## Environment Setup
Python 3.12+ required

### Dependencies
#### Option 1. Install with uv (recommended)
```bash
uv sync --python 3.12
```
For uv installation and usage, please refer to the [official documentation](https://docs.astral.sh/uv/getting-started/).

#### Option 2. Install with pip
```bash
pip install -r requirements.txt
```

### Experiment Tracking
We use [Weights & Biases (W&B)](https://wandb.ai/site) for monitoring training logs. If you use W&B, please log in to your W&B account using the following command:
```bash
wandb login
```

## Usage
Here, we explain the procedure for finetuning [kyutai/moshiko-pytorch-bf16](https://huggingface.co/kyutai/moshiko-pytorch-bf16). Note that in this instruction, we demonstrate using uv. If you don't use uv, replace `uv run` with `python` or other commands suitable for your environment.

### 1. Data Preparation
Build a dataset for finetuning by tokenizing spoken dialogue data and its transcription text into discrete tokens. We use a small amount of sample data extracted from [SpokenWOZ](https://arxiv.org/abs/2305.13040) as an example. For details, see [`data/spokenwoz_sample`](data/spokenwoz_sample).

#### 1.1. Audio Tokenization
Spoken dialogue data should be wav files with two speakers (A and B) separated into respective channels. The left and right channel should contain speaker A's and B's audio, respectively. One wav file should contain one dialogue data.
For specific examples of the data, refer to `data/spokenwoz_sample/audio/*.wav`.

Use the following script to convert all wav files in the directory to discrete tokens using the Mimi encoder:
```bash
uv run -m tools.tokenize_audio \
    --audio_dir data/spokenwoz_sample/audio \
    --output_dir data/spokenwoz_sample/tokenized_audio
```

This will create `data/spokenwoz_sample/tokenized_audio/*.npz`. Each npz file contains audio tokens for A and B:
```python
>>> import numpy as np
>>> npz = np.load("data/spokenwoz_sample/tokenized_audio/SNG0072.npz")
>>> npz["A"].shape[0]
8 # levels of residual vector quantization
>>> npz["A"].shape[1]
1271 # frames of audio token streams (12.5Hz)
>>> npz["A"].shape == npz["B"].shape
True
```

#### 1.2. Text Tokenization
Text data (json) are word-level transcriptions with timestamps. One json file should contain one dialogue data, i.e., the transcriptions of both speakers (A and B). The format of the data is as follows:
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
As shown in this example, each element includes speaker (`speaker`), word (`word`), start time (`start`), and end time (`end`). `start` and `end` represent the seconds in the corresponding wav file. For specific examples of the data, see `data/spokenwoz_sample/text/*.json`. If your dataset does not include word-level timestamps, you need to create them using other libraries with [forced alignment](https://pytorch.org/audio/stable/tutorials/forced_alignment_tutorial.html), such as [WhisperX](https://github.com/m-bain/whisperX). For detailed usage of these libraries, please refer to their respective repositories.

Use the following script to convert the text for all json files in the directory to tokens:
```bash
uv run -m tools.tokenize_text \
    --word_transcript data/spokenwoz_sample/text \
    --output_dir data/spokenwoz_sample/tokenized_text
```

This will create `data/spokenwoz_sample/tokenized_text/*.npz`. Each npz file contains text tokens for A and B:
```python
>>> import numpy as np
>>> npz = np.load("data/spokenwoz_sample/tokenized_audio/SNG0072.npz")
>>> npz["A"].shape
(1271,) # frames of text token stream (12.5Hz)
```

#### Tips: Use Other Text Tokenizers
If you want to use a text tokenizer other than the one provided by Kyutai, specify it by using the `--text_tokenizer_repo` and `--text_tokenizer_name` arguments. Note that only SentencePiece models are supported as tokenizers. For example, J-Moshi used the following settings:
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
> With the change of tokenizer, you may need to change the ID of the padding token (`--text_padding_id`) and the ID of the end of padding token (`--end_of_text_padding_id`). Also, for languages such as Japanese and Chinese, where there are no spaces between words, use the `--no_whitespace_before_word` flag.


#### 1.3. Concatenation of Audio and Text Tokens
Concatenate sequences of audio token and text token to create a ready-to-use dataset for finetuning.
```bash
uv run -m tools.prepare_dataset \
    --tokenized_text_dir data/spokenwoz_sample/tokenized_text \
    --tokenized_audio_dir data/spokenwoz_sample/tokenized_audio \
    --output_prefix processed_data/spokenwoz_sample/train
```
This command creates `processed_data/spokenwoz_sample/train-001-of-001.parquet`. One parquet file can contain up to 100,000 dialogues. The structure of the dataset is as follows:
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
Here, we initialize and edit the model weights published by Kyutai for finetuning. This process mainly includes the following operations:
- (If you changed the text tokenizer) Initialize the embedding table for text tokens
    - Use the `--init_text_embeddings` flag.
    - **Currently, changing the vocabulary size is not supported. Be sure to use a tokenizer with the same vocabulary size.**
- Add modules to the Depth Transformer to output the user's speech stream
    - Use the `--extend_modules_for_user_stream` flag
- Modify (Monkey patch) some modules in the Transformer for training with DeepSpeed Zero 3

Run the following command to initialize the model:
```bash
uv run -m tools.init_moshi_for_ft \
    --moshi_lm_repo kyutai/moshiko-pytorch-bf16 \
    --save_dir init_models/moshiko-both_streams-float32 \
    --model_dtype float32 \
    --extend_modules_for_user_stream
```
This saves the initialized model (`model.safetensors`) and its configuration file (`moshi_lm_kwargs.json`) to `init_models/moshiko-both_streams-float32`.
If you're using a GPU that doesn't support bfloat16, specify `--model_dtype float32`.


### 3. Training
Run finetuning using the dataset and initialized model created in Sections 1 and 2. Basically, use the 🤗 Accelerate launcher and run with the desired number of processes:
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
Currently, training without DeepSpeed is not supported, so be sure to specify `--use_deepspeed` and `--deepspeed_config_file`. Batch size, learning rate, weights of loss for each token, etc. can be specified as arguments to `finetune.py`. Use `uv run finetune.py --help` for details.

> [!NOTE]
> For specific execution commands, refer to `examples/finetune_accelerate.sh`.

Checkpoints during training are saved as DeepSpeed state files. To use them for later inference, convert the state files to safetensors format using the following script:
```bash
uv run -m tools.zero_to_fp32 \
    output/moshiko-finetuned/step_10000 \
    output/moshiko-finetuned/step_10000_fp32 \
    --moshi_lm_kwargs_path init_models/moshiko-both_streams-float32/moshi_lm_kwargs.json
```
This saves the model weights `model.safetensors` and its configuration file `moshi_lm_kwargs.json` to `output/moshiko-finetuned/step_10000_fp32`.

#### Tips: Multi-Node Training
For multi-node training, run the `accelerate launch` command with additional arguments `--machine_rank`, `--main_process_ip`, `--main_process_port`, and `--deepspeed_multinode_launcher standard` on all nodes. For more detailed usage of multi-node training with 🤗 Accelerate, see the [official documentation](https://huggingface.co/docs/accelerate/basic_tutorials/launch#multi-node-training).

> [!NOTE]
> For controlling multi-node processing with OpenMPI's `mpirun`, refer to `examples/finetune_mpi_accelerate.sh`. If you want to launch all processes with `mpirun` without using `accelerate launch`, refer to `examples/finetune_mpi.sh`.


## Inference
You can generate spoken dialogues and have real-time conversations using the finetuned model.


### 1. Prompted Dialogue Continuation
You can generate a continuation from a few seconds of prompt contained in the dataset created in [Data Preparation](#1-data-preparation). Run the following command to generate the continuation:
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
Specify both the length of the prompt and the generation using `--prompt_length` and `--generation_length` respectively. For each sample in the dataset, the part from the beginning to the length specified by `--prompt_length` in the sample is input to the model. The unit is one frame of Mimi (80ms). Other settings such as batch size and temperature parameter can be specified as arguments to `generate.py`. Use `uv run generate.py --help` for details.

> [!NOTE]
> For specific execution commands, refer to `examples/generate_accelerate.sh`.

#### Tips: Decode generated audio tokens to wav
The generated tokens are saved as npy files for each dialogue in `generated_tokens` under the directory specified by `--output_dir` in the arguments of generate.py. You can convert these npy files to wav files using the following script:
```bash
uv run -m tools.decode_tokens \
    --tokens_dir "${model_dir}/continuation/generated_tokens" \
    --output_dir "${model_dir}/continuation/generated_wavs"
```
This saves wav files for each dialogue. Each wav file contains the system's speech in the left channel and the user's speech in the right channel.


### 2. Interactive Demo
You can have real-time conversations with the finetuned model using the web app (`moshi.server`) of the [moshi library](https://github.com/kyutai-labs/moshi). Follow these steps:

#### 2.1 Cleaning the finetuned model
To load the model in `moshi.server`, you first need to revert the modules edited for finetuning in [Model Initialization](#2-model-initialization) back to the original Moshi model format. Specifically, this includes:
- Removing the modules for the user's speech stream in the Depth Transformer
- Fixing the modules edited for DeepSpeed Zero 3

Run the following command to revert the model to its original format:
```bash
uv run -m tools.clean_moshi \
    --moshi_ft_dir output/moshiko-finetuned/step_10000_fp32 \
    --save_dir output/moshiko-finetuned/step_10000_cleaned \
    --model_dtype float32 \
    --remove_modules_for_user_stream
```
This saves the model in its original format (`model.safetensors`) and its configuration file (`moshi_lm_kwargs.json`) to `output/moshiko-finetuned/step_10000_cleaned`.

#### 2.2 Running the server
Run the following command to start the server:
```bash
uv run -m moshi.server \
    --moshi-weight output/moshiko-finetuned/step_10000_cleaned
```
By default, the server starts at `http://localhost:8998`. You can start a conversation by accessing this URL in your browser. For detailed usage, refer to the moshi repository.


## License
This repository is provided under the [Apache 2.0 License](LICENSE). The SpokenWOZ sample data included in `data/spokenwoz_sample` is provided under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).


## Citation
If you use this repository, please cite the following:
```bibtex
@article{ohashi2025towards,
    title={Towards a Japanese Full-duplex Spoken Dialogue System},
    author={Ohashi, Atsumoto and Iizuka, Shinya and Jiang, Jingjing and Higashinaka, Ryuichiro},
    journal={arXiv preprint arXiv:2506.02979},
    year={2025}
}
```
