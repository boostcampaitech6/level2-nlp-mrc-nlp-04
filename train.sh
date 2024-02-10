python ./code/train.py \
--do_train\
--per_device_train_batch_size 16 \ 
--per_device_eval_batch_size 16 \
--num_train_epochs 3 \
--output_dir ./models/train_dataset \
--overwrite_output_dir \ 
--overwrite_cache 