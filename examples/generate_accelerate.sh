export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

model_dir="output/moshiko-finetuned/step_10000_fp32"
eval_data_files="path/to/proccessed_data/test-*.parquet"

prompt_len=125
gen_len=250

uv run accelerate launch \
    --num_machines 1 \
    --num_processes 4 \
    generate.py \
        --output_dir ${model_dir}/continuation \
        --model_dir ${model_dir} \
        --model_dtype float16 \
        --eval_data_files "${eval_data_files}" \
        --example_length $((${prompt_len} + ${gen_len})) \
        --prompt_length ${prompt_len} \
        --generation_length ${gen_len} \
        --temperature 0.8 \
        --seed 20260825
