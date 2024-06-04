DATA_DIR='./data/risawoz'
export OMP_NUM_THREADS=4

CUDA_VISIBLE_DEVICES=0 python3 ./main.py \
    --model_name_or_path './pretrained_model/Randeng-T5-77M' \
    --save_total_limit 1 \
    --do_train \
    --train_file "$DATA_DIR/train.json" \
    --validation_file "$DATA_DIR/dev.json" \
    --test_file "$DATA_DIR/test.json" \
    --cache_dir './cache_dir' \
    --output_dir './output' \
    --per_device_train_batch_size=4 \
    --learning_rate=1e-4 \
    --preprocessing_num_workers=2 \
    --warmup_ratio=0.1 \
    --text_column="dialogue" \
    --summary_column="state" \
    --overwrite_output_dir \
    --num_train_epochs 1 \
    --logging_steps 50