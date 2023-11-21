mkdir saved_models
python run_gcb.py \
    --output_dir=saved_models \
    --config_name=microsoft/graphcodebert-base \
    --model_name_or_path=microsoft/graphcodebert-base \
    --tokenizer_name=microsoft/graphcodebert-base \
    --do_train \
    --train_data_file=../dataset/tl/train \
    --eval_data_file=../dataset/tl/valid \
    --test_data_file=../dataset/tl/test \
    --num_train_epochs 3 \
    --max_source_length 256 \
    --max_target_length 128 \
    --data_flow_length 128 \
    --train_batch_size 64 \
    --eval_batch_size 64 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456 2>&1| tee saved_models/train.log