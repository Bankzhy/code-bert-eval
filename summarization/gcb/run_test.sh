source=java
pretrained_model=microsoft/graphcodebert-base
batch_size=64
output_dir=saved_models/$source/
dev_file=../dataset/tl/valid
test_file=../dataset/tl/test
load_model_path=$output_dir/checkpoint-best-bleu/pytorch_model.bin #checkpoint for test
source_length=256
target_length=128
beam_size=10

python run.py \
--do_test \
--model_type roberta \
--source_lang $source \
--model_name_or_path $pretrained_model \
--tokenizer_name microsoft/graphcodebert-base \
--config_name microsoft/graphcodebert-base \
--load_model_path $load_model_path \
--dev_filename $dev_file \
--test_filename $test_file \
--output_dir $output_dir \
--max_source_length $source_length \
--max_target_length $target_length \
--beam_size $beam_size \
--eval_batch_size $batch_size 2>&1| tee $output_dir/test.log