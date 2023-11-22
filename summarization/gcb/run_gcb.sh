source=java
lr=1e-4
batch_size=32
beam_size=10
source_length=256
target_length=128
output_dir=saved_models/$source/
train_file=../dataset/tl/train
dev_file=../dataset/tl/valid
epochs=60
pretrained_model=microsoft/graphcodebert-base

mkdir -p $output_dir
python run_gcb.py \
--do_train \
--do_eval \
--model_type roberta \
--source_lang $source \
--model_name_or_path $pretrained_model \
--tokenizer_name microsoft/graphcodebert-base \
--config_name microsoft/graphcodebert-base \
--train_filename $train_file \
--dev_filename $dev_file \
--output_dir $output_dir \
--max_source_length $source_length \
--max_target_length $target_length \
--beam_size $beam_size \
--train_batch_size $batch_size \
--eval_batch_size $batch_size \
--learning_rate $lr \
--num_train_epochs $epochs 2>&1| tee $output_dir/train.log \
--patience 6