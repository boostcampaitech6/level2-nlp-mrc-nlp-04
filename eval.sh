python ./code/train.py \
 --do_eval \
 --per_device_eval_batch_size 16 \
 --seed 2023 \
 --model_name_or_path ./models/train_dataset/ \
 --dataset_name ./data/train_dataset/ \
 --output_dir ./outputs/train_dataset \
 --overwrite_output_dir \
 --overwrite_cache