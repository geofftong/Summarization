#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: geofftong
# Mail: geofftong@tencent.com
# Created Time: 2019/4/10 6:07 PM

# !/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: geofftong
# Mail: geofftong@tencent.com
# Created Time: 2019/4/10 12:01 PM

import codecs
import json
import os
from collections import OrderedDict

import pandas as pd
from pyrouge import Rouge155


def convert_word_to_id(data_dir="hyp", vocab_path="vocab.txt"):
    """
    for chinese word and special punctuation
    :param data_dir:
    :param vocab_path: vocabulary path
    :return:
    """
    vocab = dict()
    with codecs.open(vocab_path, encoding="utf8") as f:
        for idx, line in enumerate(f):
            vocab[line.strip()] = idx
    if not os.path.exists(data_dir + "_id"):
        os.mkdir(data_dir + "_id")
    for file_name in os.listdir(data_dir):
        with codecs.open(os.path.join(data_dir, file_name), encoding="utf8") as f:
            with codecs.open(os.path.join(data_dir + "_id", file_name), "w", encoding="utf8") as fw:
                for idx, line in enumerate(f):
                    convert_list = list()
                    for char in line.strip():
                        if char not in vocab:
                            char = '[UNK]'
                        convert_list.append(str(vocab[char]))
                    fw.write(" ".join(convert_list) + "\n")


def remove_broken_files(hyp, ref):
    error_id = []
    for f in os.listdir(ref):
        try:
            open(os.path.join(ref, f)).read()
        except:
            error_id.append(f)
    for f in os.listdir(hyp):
        try:
            open(os.path.join(hyp, f)).read()
        except:
            error_id.append(f)
    error_set = set(error_id)
    for f in error_set:
        os.remove(os.path.join(ref, f))
        os.remove(os.path.join(hyp, f))


def rouge(hyp, ref):
    r = Rouge155("/Users/geofftong/Documents/Git/evaluation/ROUGE-RELEASE-1.5.5")
    r.system_dir = hyp
    r.model_dir = ref
    r.system_filename_pattern = '(\d+).txt'
    r.model_filename_pattern = '#ID#.txt'
    output = r.convert_and_evaluate()
    print(output)


def split_prediction_to_files(prediction_file, raw_test_file, output="display.txt", output_excel="display.xlsx",
                              ref="ref", hyp="hyp", export_excel=False):
    predict_dict = dict()
    gold_dict = dict()
    gold_label = dict()
    predict_label = dict()
    if not os.path.exists(ref):
        os.mkdir(ref)
    if not os.path.exists(hyp):
        os.mkdir(hyp)
    with codecs.open(prediction_file, encoding="utf8") as f:
        for idx, line in enumerate(f):
            qid_pid_sent, gold, pred = line.strip().split(" ")
            tokens = qid_pid_sent.split("-")
            qid, pid, sent = tokens[0], tokens[1], "-".join(tokens[2:])
            if qid not in gold_dict:
                gold_dict[qid] = list()
                gold_label[qid] = list()
            if qid not in predict_dict:
                predict_dict[qid] = list()
                predict_label[qid] = list()
            # if gold == '1':
            if int(gold) > 0:
                gold_dict[qid].append(sent)
                gold_label[qid].append(1)
            else:
                gold_label[qid].append(0)
            # if pred == '1':
            if int(pred) > 0:
                predict_dict[qid].append(sent)
                predict_label[qid].append(1)
            else:
                predict_label[qid].append(0)

    assert len(predict_dict) == len(gold_dict)
    assert len(predict_label) == len(gold_label)

    for qid in gold_dict:
        with codecs.open(os.path.join(ref, str(qid) + '.txt'), 'w', encoding="utf8") as f:
            for sent in gold_dict[qid]:
                f.write(sent + "\n")

        with codecs.open(os.path.join(hyp, str(qid) + '.txt'), 'w', encoding="utf8") as f2:
            for sent in predict_dict[qid]:
                f2.write(sent + "\n")

    json_data_list = list()
    ref_len, pred_len = 0, 0
    with codecs.open(output, "w", encoding="utf8") as fw:
        with codecs.open(raw_test_file, encoding="utf8") as f:
            for idx, line in enumerate(f):
                item = json.loads(line.strip())
                qid = item["id"]  # + item["category"]
                fw.write(str(qid) + ".###" + "###" + item["title"] + "\n" + "contents: " + item[
                    "content"] + "\n" + "reference: " + "".join(
                    gold_dict[str(qid)]) + "\n" + "prediction:" + "[SEP]".join(
                    predict_dict[str(qid)]) + "\n\n")
                json_data = {"title": item["title"], "id": str(qid), "url": item["url"], "content": item["content"],
                             "anno_result": "".join(gold_dict[str(qid)]),
                             "prediction": "".join(predict_dict[str(qid)])}  # , "category": item["category"]
                json_data_list.append(
                    json.loads(json.dumps(json_data, ensure_ascii=False), object_pairs_hook=OrderedDict))
                ref_len += len("".join(gold_dict[str(qid)]))
                pred_len += len("".join(predict_dict[str(qid)]))

    if export_excel:
        df = pd.DataFrame()  # 最后转换得到的结果
        for line in json_data_list:
            df1 = pd.DataFrame([line])
            df = df.append(df1)
        # 在excel表格的第1列写入, 不写入index
        df.to_excel(output_excel, sheet_name='Data', startcol=0, index=False)
    return ref_len * 1.0 / len(json_data_list), pred_len * 1.0 / len(json_data_list)


if __name__ == '__main__':
    hyp_dir, ref_dir = "hyp", "ref"
    vocab_path = "../../data_dir/model_zh/vocab.txt"

    raw_test_file = "../../data_dir/news/summary/第一期/news_ch.test"  # 0903/p2.news.test
    prediction_file = "result/final_results_xlsum_3500_0.3"  # p1_dim1r_5_1015 p2_xlnet_4_1024_beam_1
    output_file = "result/final_results_xlsum_3500_0.3.display"
    output_excel_file = "result/final_results_xlsum_3500_0.3.xlsx"

    avg_ref, avg_pred = split_prediction_to_files(prediction_file=prediction_file,
                                                  raw_test_file=raw_test_file, output=output_file,
                                                  output_excel=output_excel_file,
                                                  ref=ref_dir, hyp=hyp_dir, export_excel=False)  # False
    convert_word_to_id(data_dir=hyp_dir, vocab_path=vocab_path)
    convert_word_to_id(data_dir=ref_dir, vocab_path=vocab_path)
    # remove_broken_files(hyp=hyp_dir + "_id", ref=ref_dir + "_id")

    rouge(hyp=hyp_dir + "_id", ref=ref_dir + "_id")
    os.system("rm -rf %s" % hyp_dir)
    os.system("rm -rf %s" % ref_dir)
    os.system("rm -rf %s" % (hyp_dir + "_id"))
    os.system("rm -rf %s" % (ref_dir + "_id"))
    print("len of avg ref: %f, len of avg pred: %f" % (avg_ref, avg_pred))
    print("well done!")
