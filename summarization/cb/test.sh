lang=java #programming language
beam_size=10
batch_size=128
source_length=256
target_length=128
output_dir=model/$lang
data_dir=../dataset
dev_file=$data_dir/tl/valid
test_file=$data_dir/tl/test
test_model=$output_dir/checkpoint-best-bleu/pytorch_model.bin #checkpoint for test

python run.py --do_test --model_type roberta --model_name_or_path microsoft/codebert-base --load_model_path $test_model --dev_filename $dev_file --test_filename $test_file --output_dir $output_dir --max_source_length $source_length --max_target_length $target_length --beam_size $beam_size --eval_batch_size $batch_size