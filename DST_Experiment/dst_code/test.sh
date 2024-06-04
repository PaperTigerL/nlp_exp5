DATA_DIR='./data/risawoz'
export OMP_NUM_THREADS=4

CUDA_VISIBLE_DEVICES=0 python3 ./main.py \
    --model_name_or_path './output' \
    --do_predict \
    --train_file "$DATA_DIR/train.json" \
    --validation_file "$DATA_DIR/dev.json" \
    --test_file "$DATA_DIR/test.json" \
    --cache_dir './cache_dir' \
    --output_dir './output/test_result' \
    --predict_with_generate \
    --per_device_train_batch_size=8 \
    --text_column="dialogue" \
    --summary_column="state"