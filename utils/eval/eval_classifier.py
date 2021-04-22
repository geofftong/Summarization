#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: geofftong
# Mail: geofftong@tencent.com
# Created Time: 2019/9/18 4:40 PM
import os
import json
import codecs
import sys

reload(sys)
sys.setdefaultencoding('utf-8')

import os

import pandas as pd


def write_to_excel(raw_data_path, pred_data_path, output_path="test.result.analysis.xlsx"):
    pred_list = list()
    raw_data_list = list()
    all_data_list = list()
    with open(pred_data_path, encoding="utf8") as f:
        for idx, line in enumerate(f):
            predict_label, score = line.strip().split(" ")
            pred_list.append((predict_label, score))  # predict_label, score
    with open(raw_data_path, encoding="utf8") as f:
        for idx, line in enumerate(f):
            item = line.strip().split("\t")
            raw_data_list.append((item[4], item[7], item[8], item[9]))  # title url content gold_label

    for idx, item in enumerate(pred_list):
        all_data_list.append(
            {"title": raw_data_list[idx][0], "url": raw_data_list[idx][1], "content": raw_data_list[idx][2],
             "reference": raw_data_list[idx][3], "prediction": item[0], "score": item[1]})

    df = pd.DataFrame()  # 最后转换得到的结果
    for line in all_data_list:
        df1 = pd.DataFrame([line])
        df = df.append(df1)

    # 在excel表格的第1列写入, 不写入index
    df.to_excel(output_path, sheet_name='Data', startcol=0, index=False)


def fasttext_eval(ref_data, pred_data):
    ref_list = list()
    pred_list = list()
    data_list = list()
    with open(ref_data, encoding="utf8") as f:
        for idx, line in enumerate(f):
            label, tokens = line.strip().split("\t")
            ref_list.append([label, tokens])
    with open(pred_data, encoding="utf8") as f:
        for idx, line in enumerate(f):
            predict_label, score = line.strip().split(" ")
            pred_list.append(predict_label)
    assert len(ref_list) == len(pred_list)
    with open("temp.result", "w", encoding="utf8") as fw:
        for idx, item in enumerate(ref_list):
            fw.write("".join(item[1].split()[:10]) + " " + item[0] + " " + pred_list[idx] + "\n")
            data_list.append({"content": "".join(item[1].split()[:10])})
    os.system('perl conlleval.pl -r < {} 2>&1 | tee {}'.format("temp.result", "perl.eval.result"))
    # os.system('rm -rf {}'.format("temp.result"))


def bert_eval(ref_data, pred_data):
    ref_list = list()
    pred_list = list()
    with codecs.open(ref_data, encoding="utf8") as f:
        for idx, line in enumerate(f):
            item = json.loads(line.strip())
            label, tokens = item["label"], item["id"]
            ref_list.append([label, tokens])
    with codecs.open(pred_data, encoding="utf8") as f:
        for idx, line in enumerate(f):
            label_0_score, label_1_score = line.strip().split("\t")
            if float(label_0_score) > float(label_1_score):
                predict_label = "适合"
            else:
                predict_label = "不适合"
            pred_list.append(predict_label)
    assert len(ref_list) == len(pred_list)
    with codecs.open("temp.result", "w", encoding="utf8") as fw:
        for idx, item in enumerate(ref_list):
            fw.write(item[1] + " " + item[0] + " " + pred_list[idx] + "\n")
    os.system('perl conlleval.pl -r < {} 2>&1 | tee {}'.format("temp.result", "perl.eval.result"))
    # os.system('rm -rf {}'.format("temp.result"))


if __name__ == '__main__':
    # ref_data_path = "../data_dir/news/classifier/news.classifier.test"
    # pred_data_path = "../data_dir/news/classifier/news.test.result"
    # raw_data_path = "../data_dir/news/classifier/news.classfier.test.raw"
    # fasttext_eval(ref_data_path, pred_data_path)
    # write_to_excel(raw_data_path, pred_data_path, output_path="../data_dir/news/classifier/test.result.analysis.xlsx")
    ref_data_path = "../../data_dir/news/classifier/news.bert.test"
    pred_data_path = "test_results.tsv"
    bert_eval(ref_data_path, pred_data_path)
