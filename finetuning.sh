nohup \
accelerate launch \
--config_file "accelerate_config.yaml" \
--main_process_port 0 \
finetuning.py \
--model_path "/data/download-model/DeepSeek-R1-0528-Qwen3-8B" \
--dataset_path "data/train.csv" \
--output_dir "fine256" \
--checkpoint_dir "checkpoints" \
--checkpoint_interval 50 \
--num_epochs 5 \
--batch_size 1 \
--num_labels 7 \
--max_length 256 \
>> finetuning256.log \
2>&1 \
&
disown
