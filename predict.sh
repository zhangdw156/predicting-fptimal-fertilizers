export MAX_LENGTH=256
export NUM_LABELS=7
python predict.py \
--model_path "/data/download-model/DeepSeek-R1-0528-Qwen3-8B" \
--lora_path "checkpoints/checkpoint-step-161650" \
--input_file "data/test.csv" \
--output_file "submission.csv" \
--num_labels ${NUM_LABELS} \
--max_length ${MAX_LENGTH} 