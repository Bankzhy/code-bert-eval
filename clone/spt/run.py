import re
import sys
import os

curPath = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(curPath)
print("当前的工作目录：",os.getcwd())
print("python搜索模块的路径集合",sys.path)
import argparse
import logging
import random
import time

import numpy as np
import torch
import torch.nn as nn
from typing import Union, Tuple

from prettytable import PrettyTable
from torch.optim import AdamW
from torch.utils.data import SequentialSampler
from torch.utils.tensorboard import SummaryWriter
from transformers import BartConfig, get_scheduler, SchedulerType, get_linear_schedule_with_warmup

from clone.spt.args import add_args
from clone.spt.models.utils import get_elapse_time
from utils.general import count_params, human_format, layer_wise_parameters
from clone.spt import enums
from clone.spt.data.dataset import init_dataset
from models.bart import BartForClassificationAndGeneration, BartCloneModel
from data.vocab import Vocab, load_vocab, init_vocab
from torch.utils.data.dataloader import DataLoader
from data.data_collator import collate_fn
from accelerate import Accelerator
import math
from utils.early_stopping import EarlyStopping
from tqdm import tqdm

logger = logging.getLogger(__name__)

def evaluate(args, model, eval_dataloader, eval_when_training=False):
    # build dataloader
    datasets = dict()

    # eval_dataset = datasets['valid'] = init_dataset(args=args,
    #                                    mode=enums.TRAINING_MODE_FINE_TUNE,
    #                                    task=enums.TASK_CLONE_DETECTION,
    #                                    language=args.summarization_language,
    #                                    split=split)
    # eval_sampler = SequentialSampler(eval_dataset)
    # eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=4)

    # multi-gpu evaluate
    if args.n_gpu > 1 and eval_when_training is False:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataloader))
    logger.info("  Batch size = %d", args.eval_batch_size)

    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    logits = []
    y_trues = []
    for batch in eval_dataloader:
        # (source_ids, labels) = [x.to(args.device) for x in batch]
        with torch.no_grad():
            lm_loss, logit = model(batch["input_ids"], batch["attention_mask"], batch["decoder_input_ids"], batch["decoder_attention_mask"], batch["labels"])
            eval_loss += lm_loss.mean().item()
            logits.append(logit.cpu().numpy())
            y_trues.append(batch["labels"].cpu().numpy())
        nb_eval_steps += 1

    # calculate scores
    logits = np.concatenate(logits, 0)
    y_trues = np.concatenate(y_trues, 0)
    best_threshold = 0.5
    best_f1 = 0

    y_preds = logits[:, 1] > best_threshold
    from sklearn.metrics import recall_score
    recall = recall_score(y_trues, y_preds)
    from sklearn.metrics import precision_score
    precision = precision_score(y_trues, y_preds)
    from sklearn.metrics import f1_score
    f1 = f1_score(y_trues, y_preds)
    result = {
        "eval_recall": float(recall),
        "eval_precision": float(precision),
        "eval_f1": float(f1),
        "eval_threshold": best_threshold,

    }

    logger.info("***** Eval results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(round(result[key], 4)))

    return result




def run_clone(args,
        trained_model: Union[BartForClassificationAndGeneration, str] = None,
        trained_vocab: Union[Tuple[Vocab, Vocab, Vocab], str] = None,
        only_test=False):
    accelerator = Accelerator()
    logger.info('-' * 100)
    logger.info(f'Code Clone')
    # --------------------------------------------------
    # datasets
    # --------------------------------------------------
    logger.info('-' * 100)
    logger.info('Loading datasets')
    datasets = dict()
    splits = ['test'] if only_test else ['train', 'valid', 'test']
    for split in splits:
        datasets[split] = init_dataset(args=args,
                                       mode=enums.TRAINING_MODE_FINE_TUNE,
                                       task=enums.TASK_CLONE_DETECTION,
                                       language=args.summarization_language,
                                       split=split)
        logger.info(f'The size of {split} set: {len(datasets[split])}')
    if args.train_subset_ratio and 'train' in datasets:
        datasets['train'] = datasets['train'].subset(args.train_subset_ratio)
        logger.info(f'The train is trimmed to subset due to the argument: train_subset_ratio={args.train_subset_ratio}')
        logger.info('The size of trimmed train set: {}'.format(len(datasets['train'])))

    logger.info('Datasets loaded successfully')

    # --------------------------------------------------
    # vocabs
    # --------------------------------------------------
    logger.info('-' * 100)
    if trained_vocab:
        if isinstance(trained_vocab, tuple):
            logger.info('Vocabularies are passed through parameter')
            assert len(trained_vocab) == 3
            code_vocab, ast_vocab, nl_vocab = trained_vocab
        else:
            logger.info('Loading vocabularies from files')
            code_vocab = load_vocab(vocab_root=trained_vocab, name=args.code_vocab_name)
            ast_vocab = load_vocab(vocab_root=trained_vocab, name=args.ast_vocab_name)
            nl_vocab = load_vocab(vocab_root=trained_vocab, name=args.nl_vocab_name)
    else:
        logger.info('Building vocabularies')
        code_vocab = init_vocab(vocab_save_dir=args.vocab_save_dir,
                                name=args.code_vocab_name,
                                method=args.code_tokenize_method,
                                vocab_size=args.code_vocab_size,
                                datasets=[datasets['train'].codes_1, datasets['train'].codes_2],
                                ignore_case=True,
                                save_root=args.vocab_root)
        nl_vocab = init_vocab(vocab_save_dir=args.vocab_save_dir,
                              name=args.nl_vocab_name,
                              method=args.nl_tokenize_method,
                              vocab_size=args.nl_vocab_size,
                              datasets=[datasets['train'].names_1,datasets['train'].names_2],
                              ignore_case=True,
                              save_root=args.vocab_root,
                              index_offset=len(code_vocab))
        ast_vocab = init_vocab(vocab_save_dir=args.vocab_save_dir,
                               name=args.ast_vocab_name,
                               method='word',
                               datasets=[datasets['train'].asts_1, datasets['train'].asts_2],
                               save_root=args.vocab_root,
                               index_offset=len(code_vocab) + len(nl_vocab))
    logger.info(f'The size of code vocabulary: {len(code_vocab)}')
    logger.info(f'The size of nl vocabulary: {len(nl_vocab)}')
    logger.info(f'The size of ast vocabulary: {len(ast_vocab)}')
    logger.info('Vocabularies built successfully')

    # --------------------------------------------------
    # model
    # --------------------------------------------------
    logger.info('-' * 100)
    if trained_model:
        if isinstance(trained_model, BartForClassificationAndGeneration):
            logger.info('Model is passed through parameter')
            model = trained_model
        else:
            logger.info('Loading the model from file')
            config = BartConfig.from_json_file(os.path.join(trained_model, 'config.json'))
            model = BartForClassificationAndGeneration.from_pretrained(os.path.join(trained_model, 'pytorch_model.bin'),
                                                                       config=config)
    else:
        logger.info('Building the model')
        config = BartConfig(vocab_size=len(code_vocab) + len(ast_vocab) + len(nl_vocab),
                            max_position_embeddings=1024,
                            encoder_layers=args.n_layer,
                            encoder_ffn_dim=args.d_ff,
                            encoder_attention_heads=args.n_head,
                            decoder_layers=args.n_layer,
                            decoder_ffn_dim=args.d_ff,
                            decoder_attention_heads=args.n_head,
                            activation_function='gelu',
                            d_model=args.d_model,
                            dropout=args.dropout,
                            use_cache=True,
                            pad_token_id=Vocab.START_VOCAB.index(Vocab.PAD_TOKEN),
                            bos_token_id=Vocab.START_VOCAB.index(Vocab.SOS_TOKEN),
                            eos_token_id=Vocab.START_VOCAB.index(Vocab.EOS_TOKEN),
                            is_encoder_decoder=True,
                            decoder_start_token_id=Vocab.START_VOCAB.index(Vocab.SOS_TOKEN),
                            forced_eos_token_id=Vocab.START_VOCAB.index(Vocab.EOS_TOKEN),
                            max_length=args.max_code_len,
                            min_length=1,
                            num_beams=args.beam_width,
                            num_labels=2)
        model = BartForClassificationAndGeneration(config, mode=enums.MODEL_MODE_GEN)
    model.set_model_mode(enums.MODEL_MODE_GEN)
    # log model statistic
    logger.info('Trainable parameters: {}'.format(human_format(count_params(model))))
    table = layer_wise_parameters(model)
    logger.debug('Layer-wised trainable parameters:\n{}'.format(table))
    logger.info('Model built successfully')
    model = BartCloneModel(
        encoder=model,
        config=config,
        code_vocab=code_vocab,
        args=args
    )
    fa = open(os.path.join(args.output_root, 'summary.log'), 'a+')
    if args.n_gpu > 1:
        # multi-gpu training
        model = torch.nn.DataParallel(model)
    if args.local_rank in [-1, 0] and args.data_num == -1:
        summary_fn = '{}/{}'.format(args.summary_dir, '/'.join(args.output_root.split('/')[1:]))
        tb_writer = SummaryWriter(summary_fn)
    # --------------------------------------------------
    # trainer
    # --------------------------------------------------
    if not only_test:
        logger.info('-' * 100)
        logger.info('Initializing the running configurations')
        gradient_accumulation_steps = 1
        # dataloader
        dataloader = DataLoader(dataset=datasets['train'],
                                batch_size=args.batch_size,
                                collate_fn=lambda batch: collate_fn(batch,
                                                                    args=args,
                                                                    task=enums.TASK_CLONE_DETECTION,
                                                                    code_vocab=code_vocab,
                                                                    nl_vocab=nl_vocab,
                                                                    ast_vocab=ast_vocab))
        valid_dataloader = DataLoader(dataset=datasets['valid'],
                                      batch_size=args.eval_batch_size,
                                      collate_fn=lambda batch: collate_fn(batch,
                                                                          args=args,
                                                                          task=enums.TASK_CLONE_DETECTION,
                                                                          code_vocab=code_vocab,
                                                                          nl_vocab=nl_vocab,
                                                                          ast_vocab=ast_vocab))
        # optimizer
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.lr_decay_rate,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
        # Prepare everything with `accelerator`
        model, optimizer, dataloader, valid_dataloader  = accelerator.prepare(
            model, optimizer, dataloader, valid_dataloader
        )
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                    num_training_steps=args.max_steps)
        # Scheduler and math around the number of training steps.
        num_update_steps_per_epoch = math.ceil(len(dataloader) / gradient_accumulation_steps)
        max_train_steps = args.n_epoch * num_update_steps_per_epoch

        lr_scheduler = get_scheduler(name=SchedulerType.LINEAR,
                                     optimizer=optimizer,
                                     num_warmup_steps=args.warmup_steps,
                                     num_training_steps=max_train_steps)
        # early stopping
        early_stopping = EarlyStopping(patience=args.early_stop_patience, higher_better=True)
        logger.info('Running configurations initialized successfully')

        # loss
        loss_fct = torch.nn.CrossEntropyLoss()
        num_train_optimization_steps = args.n_epoch * len(dataloader)
        save_steps = max(len(dataloader) // 5, 1)
        # --------------------------------------------------
        # train
        # --------------------------------------------------
        logger.info('-' * 100)
        logger.info('Start training')

        total_batch_size = args.batch_size * accelerator.num_processes * gradient_accumulation_steps

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(datasets['train'])}")
        logger.info(f"  Num Epochs = {args.n_epoch}")
        logger.info(f"  Instantaneous batch size per device = {args.batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_train_steps}")

        progress_bar = tqdm(range(max_train_steps))
        completed_steps = 0
        global_step = 0
        tr_loss, logging_loss, avg_loss, tr_nb, tr_num, train_loss = 0.0, 0.0, 0.0, 0, 0, 0
        best_f1 = 0

        model.zero_grad()
        is_early_stop = False
        for epoch in range(args.n_epoch):
            bar = tqdm(dataloader, total=len(dataloader))
            tr_num = 0
            train_loss = 0
            model.train()
            nb_tr_examples, nb_tr_steps, tr_loss = 0, 0, 0
            for step, batch in enumerate(dataloader):
                loss, logits = model(batch["input_ids"], batch["attention_mask"], batch["decoder_input_ids"], batch["decoder_attention_mask"], batch["labels"])

                if args.n_gpu > 1:
                    loss = loss.mean()

                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                tr_loss += loss.item()

                nb_tr_examples += batch["input_ids"].size(0)
                nb_tr_steps += 1
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                if nb_tr_steps % args.gradient_accumulation_steps == 0:
                    # Update parameters
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    global_step += 1
                    train_loss = round(tr_loss * args.gradient_accumulation_steps / nb_tr_steps, 4)
                    bar.set_description("[{}] Train loss {}".format(epoch, round(train_loss, 3)))

                if (step + 1) % save_steps == 0:
                    logger.info("***** CUDA.empty_cache() *****")
                    torch.cuda.empty_cache()

                    # eval_examples, eval_data = load_and_cache_clone_data(args, args.dev_filename, pool, tokenizer,
                    #                                                      'valid', is_sample=True)

                    result = evaluate(args, model, valid_dataloader, eval_when_training=True)
                    eval_f1 = result['eval_f1']

                    if args.data_num == -1:
                        tb_writer.add_scalar('dev_f1', round(eval_f1, 4), epoch)

                    # save last checkpoint
                    last_output_dir = os.path.join(args.output_root, 'checkpoint-last')
                    if not os.path.exists(last_output_dir):
                        os.makedirs(last_output_dir)

                    if True or args.data_num == -1 and args.save_last_checkpoints:
                        model_to_save = model.module if hasattr(model, 'module') else model
                        output_model_file = os.path.join(last_output_dir, "pytorch_model.bin")
                        torch.save(model_to_save.state_dict(), output_model_file)
                        logger.info("Save the last model into %s", output_model_file)

                    if eval_f1 > best_f1:
                        not_f1_inc_cnt = 0
                        logger.info("  Best f1: %s", round(eval_f1, 4))
                        logger.info("  " + "*" * 20)
                        fa.write("[%d] Best f1 changed into %.4f\n" % (epoch, round(eval_f1, 4)))
                        best_f1 = eval_f1
                        # Save best checkpoint for best ppl
                        output_dir = os.path.join(args.output_root, 'checkpoint-best-f1')
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        if args.data_num == -1 or True:
                            model_to_save = model.module if hasattr(model, 'module') else model
                            output_model_file = os.path.join(output_dir, "pytorch_model.bin")
                            torch.save(model_to_save.state_dict(), output_model_file)
                            logger.info("Save the best ppl model into %s", output_model_file)
                    else:
                        not_f1_inc_cnt += 1
                        logger.info("F1 does not increase for %d epochs", not_f1_inc_cnt)
                        if not_f1_inc_cnt > args.early_stop_patience:
                            logger.info("Early stop as f1 do not increase for %d times", not_f1_inc_cnt)
                            fa.write("[%d] Early stop as not_f1_inc_cnt=%d\n" % (epoch, not_f1_inc_cnt))
                            is_early_stop = True
                            break

                model.train()
            if is_early_stop:
                break

            logger.info("***** CUDA.empty_cache() *****")
            torch.cuda.empty_cache()

        if args.local_rank in [-1, 0] and args.data_num == -1:
            tb_writer.close()
        #         code_hidden_states = model(
        #             input_ids=batch["input_ids"],
        #             attention_mask=batch["attention_mask"],
        #             labels=batch["input_ids"],
        #             decoder_attention_mask=batch["attention_mask"],
        #             return_dict=True,
        #             output_hidden_states=True
        #         ).decoder_hidden_states[-1]
        #
        #         code_eos_mask = batch["input_ids"].eq(code_vocab.get_eos_index())
        #         if len(torch.unique_consecutive(code_eos_mask.sum(1))) > 1:
        #             raise ValueError("All examples must have the same number of <eos> tokens.")
        #         code_vec = code_hidden_states[code_eos_mask, :].view(
        #             code_hidden_states.size(0), -1, code_hidden_states.size(-1))[:, -1, :]
        #         code_vec = torch.nn.functional.normalize(code_vec, p=2, dim=1)
        #
        #         nl_hidden_states = model(
        #             input_ids=batch["nl_input_ids"],
        #             attention_mask=batch["nl_attention_mask"],
        #             labels=batch["nl_input_ids"],
        #             decoder_attention_mask=batch["nl_attention_mask"],
        #             return_dict=True,
        #             output_hidden_states=True
        #         ).decoder_hidden_states[-1]
        #         nl_eos_mask = batch["nl_input_ids"].eq(nl_vocab.get_eos_index())
        #         if len(torch.unique_consecutive(nl_eos_mask.sum(1))) > 1:
        #             raise ValueError("All examples must have the same number of <eos> tokens.")
        #         nl_vec = nl_hidden_states[nl_eos_mask, :].view(
        #             nl_hidden_states.size(0), -1, nl_hidden_states.size(-1))[:, -1, :]
        #         nl_vec = torch.nn.functional.normalize(nl_vec, p=2, dim=1)
        #
        #         # calculate scores and loss
        #         scores = torch.einsum("ab,cb->ac", nl_vec, code_vec)
        #         loss = loss_fct(scores * 20, torch.arange(batch["input_ids"].size(0), device=scores.device))
        #
        #         loss = loss / gradient_accumulation_steps
        #         accelerator.backward(loss)
        #
        #         if step % args.logging_steps == 0 and step != 0:
        #             logger.info({
        #                 "global_step": completed_steps,
        #                 "epoch": completed_steps / num_update_steps_per_epoch,
        #                 "loss": loss.item(),
        #             })
        #
        #         if step % gradient_accumulation_steps == 0 or step == len(dataloader) - 1:
        #             # Gradient clipping
        #             if args.grad_clipping_norm is not None and args.grad_clipping_norm > 0:
        #
        #                 if hasattr(optimizer, "clip_grad_norm"):
        #                     # Some optimizers (like the sharded optimizer) have a specific way to do gradient clipping
        #                     optimizer.clip_grad_norm(args.grad_clipping_norm)
        #                 elif hasattr(model, "clip_grad_norm_"):
        #                     # Some models (like FullyShardedDDP) have a specific way to do gradient clipping
        #                     model.clip_grad_norm_(args.grad_clipping_norm)
        #                 else:
        #                     # Revert to normal clipping otherwise, handling Apex or full precision
        #                     nn.utils.clip_grad_norm_(model.parameters(), args.grad_clipping_norm)
        #
        #             optimizer.step()
        #             lr_scheduler.step()
        #             optimizer.zero_grad()
        #             progress_bar.update(1)
        #             completed_steps += 1
        #
        #         if completed_steps >= max_train_steps:
        #             break
        #
        #     model.eval()
        #     logger.info("***** Running evaluation *****")
        #     logger.info("  Num queries = %d", len(datasets['valid']))
        #     logger.info("  Num codes = %d", len(datasets['codebase']))
        #     logger.info("  Batch size = %d", args.eval_batch_size)
        #     valid_result = run_eval(
        #         args=args,
        #         model=model,
        #         query_dataloader=valid_dataloader,
        #         codebase_dataloader=codebase_dataloader,
        #         code_vocab=code_vocab,
        #         nl_vocab=nl_vocab,
        #         split="valid",
        #         epoch=epoch
        #     )
        #     logger.info(valid_result)
        #     mrr = valid_result['valid_mrr']
        #     early_stopping(score=mrr, model=model.state_dict(), epoch=epoch)
        #     if early_stopping.early_stop:
        #         break
        #
        # logger.info('Training finished')
        fa.close()

    if args.do_test:
        t0 = time.time()
        logger.info("  " + "***** Testing *****")
        logger.info("  Batch size = %d", args.eval_batch_size)

        for criteria in ['best-f1']:
            file = os.path.join(args.output_dir, 'checkpoint-{}/pytorch_model.bin'.format(criteria))
            logger.info("Reload model from {}".format(file))
            model.load_state_dict(torch.load(file))

            if args.n_gpu > 1:
                # multi-gpu training
                model = torch.nn.DataParallel(model)

            # eval_examples, eval_data = load_and_cache_clone_data(args, args.test_filename, pool, tokenizer, 'test',
            #                                                      False)
            test_dataloader = DataLoader(dataset=datasets['test'],
                                         batch_size=args.eval_batch_size,
                                         collate_fn=lambda batch: collate_fn(batch,
                                                                             args=args,
                                                                             task=enums.TASK_CLONE_DETECTION,
                                                                             code_vocab=code_vocab,
                                                                             nl_vocab=nl_vocab,
                                                                             ast_vocab=ast_vocab))

            result = evaluate(args, model, test_dataloader)
            logger.info("  test_f1=%.4f", result['eval_f1'])
            logger.info("  test_prec=%.4f", result['eval_precision'])
            logger.info("  test_rec=%.4f", result['eval_recall'])
            logger.info("  " + "*" * 20)

            fa.write("[%s] test-f1: %.4f, precision: %.4f, recall: %.4f\n" % (
                criteria, result['eval_f1'], result['eval_precision'], result['eval_recall']))
            if args.res_fn:
                with open(args.res_fn, 'a+') as f:
                    f.write('[Time: {}] {}\n'.format(get_elapse_time(t0), file))
                    f.write("[%s] f1: %.4f, precision: %.4f, recall: %.4f\n\n" % (
                        criteria, result['eval_f1'], result['eval_precision'], result['eval_recall']))
    fa.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.register('type', 'bool', lambda v: v.lower() in ['yes', 'true', 't', '1', 'y'])

    add_args(parser)

    main_args = parser.parse_args()

    # define and make dirs
    # Root directory for the output of this run
    main_args.output_root = os.path.join(
        'outputs',
        '{}_{}'.format(main_args.model_name, time.strftime('%Y%m%d_%H%M%S', time.localtime())))
    # Root for outputs during pre-training
    main_args.pre_train_output_root = os.path.join(main_args.output_root, 'pre_train')
    # Root for saving checkpoints
    main_args.checkpoint_root = os.path.join(main_args.output_root, 'checkpoints')
    # Root for saving models
    main_args.model_root = os.path.join(main_args.output_root, 'models')
    # Root for saving vocabs
    main_args.vocab_root = os.path.join(main_args.output_root, 'vocabs')
    # Rot for tensorboard
    main_args.tensor_board_root = os.path.join(main_args.output_root, 'runs')
    for d in [main_args.checkpoint_root, main_args.model_root, main_args.vocab_root, main_args.tensor_board_root,
              main_args.dataset_save_dir, main_args.vocab_save_dir]:
        if not os.path.exists(d):
            os.makedirs(d)

    # cuda and parallel
    if main_args.cuda_visible_devices is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = main_args.cuda_visible_devices
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    main_args.use_cuda = torch.cuda.is_available()
    main_args.parallel = torch.cuda.device_count() > 1

    # Setup CUDA, GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main_args.n_gpu = torch.cuda.device_count()
    main_args.device = device

    # set random seed
    if main_args.random_seed > 0:
        random.seed(main_args.random_seed)
        np.random.seed(main_args.random_seed)
        torch.manual_seed(main_args.random_seed)
        torch.cuda.manual_seed_all(main_args.random_seed)

    # logging, log to both console and file, debug level only to file
    logger = logging.getLogger()
    logger.setLevel(level=logging.DEBUG)

    console = logging.StreamHandler()
    console.setLevel(level=logging.INFO)
    logger.addHandler(console)

    file = logging.FileHandler(os.path.join(main_args.output_root, 'info.log'))
    file.setLevel(level=logging.DEBUG)
    formatter = logging.Formatter('[%(asctime)s | %(filename)s | line %(lineno)d] - %(levelname)s: %(message)s')
    file.setFormatter(formatter)
    logger.addHandler(file)

    # log command and configs
    logger.debug('COMMAND: {}'.format(' '.join(sys.argv)))

    config_table = PrettyTable()
    config_table.field_names = ["Configuration", "Value"]
    config_table.align["Configuration"] = "l"
    config_table.align["Value"] = "l"
    for config, value in vars(main_args).items():
        config_table.add_row([config, str(value)])
    logger.debug('Configurations:\n{}'.format(config_table))

    run_clone(main_args)