#!/usr/bin/env bash
# go to this directory$
#export YARD_HOME=/mnt/yardcephfs/mmyard/g_wxg_sd_prc/geofftong
#cd ${YARD_HOME}/summarization/car_news_sum0827
#umask 0
export CUDA_VISIBLE_DEVICES=3
SQUAD_DIR='data_dir/preprocessed'
BERT_DIR='data_dir/model_zh'

python run_squad_dim1_ngram.py \
--vocab_file=${BERT_DIR}/vocab.txt \
--bert_config_file=${BERT_DIR}/bert_config.json \
--init_checkpoint=${BERT_DIR}/bert_model.ckpt \
--do_train=True \
--do_eval=True \
--do_predict=False \
--train_file=${SQUAD_DIR}/news.train.2875.expand.squad \
--dev_file=${SQUAD_DIR}/news.dev.99.expand.squad \
--predict_file=${SQUAD_DIR}/p2.news.squad.test \
--raw_predict_file=${SQUAD_DIR}/p2.news.test \
--train_batch_size=8 \
--predict_batch_size=16 \
--learning_rate=3e-5 \
--num_train_epochs=20.0 \
--max_seq_length=504 \
--max_query_length=128 \
--doc_stride=256 \
--max_extract_turns=8 \
--beam_sizes=1 \
--output_dir=squad_output \
--eval_every_steps=500 \
--best_model_dir=best_model_dir \
--version_2_with_negative=True \
--do_export=True \
--export_dir=squad_output/pb_model