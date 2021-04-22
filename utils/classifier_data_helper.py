#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: geofftong
# Mail: geofftong@tencent.com
# Created Time: 2019/9/18 11:15 AM
import json
import random

import numpy as np
import pandas as pd

random.seed(16)


def split_data(raw_data, output_pref):
    """
    split data to train/dev/test
    :return:
    """
    train_data = output_pref + ".train"
    test_data = output_pref + ".test"
    random.shuffle(raw_data)
    with open(train_data, "w", encoding="utf8") as fw1:
        with open(test_data, "w", encoding="utf8") as fw2:
            with open(train_data + ".raw", "w", encoding="utf8") as fw3:
                with open(test_data + ".raw", "w", encoding="utf8") as fw4:
                    for idx, (line, item) in enumerate(raw_data):
                        if idx < 1000:
                            fw2.write(line + "\n")
                            fw4.write("\t".join([str(i) for i in item]) + "\n")
                        else:
                            fw1.write(line + "\n")
                            fw3.write("\t".join([str(i) for i in item]) + "\n")


def convert_excel_data(excel_data, output_prefix):
    df = pd.read_excel(excel_data)  # 13
    data_list = list()
    for idx, item in enumerate(np.array(df).tolist()):
        category = item[2]
        content = item[4]
        title = item[8]
        label = item[9]
        if category is np.nan or content is np.nan or title is np.nan or label is np.nan:
            print("error in ", idx)
            continue
        if label not in ["适合", "不适合"]:
            print("error: ", label)
            continue
        new_line = "__label__" + label + "\t" + " ".join([_ for _ in category]) + " " + " ".join(
            [_ for _ in title]) + " " + " ".join([_ for _ in content])
        data_list.append((new_line, item))
    print("len of raw_data: %d" % len(data_list))
    split_data(data_list, output_prefix)


def convert_excel_data2(excel_data, raw_data, output_data="news.classfier.train.expand"):
    df = pd.read_excel(excel_data)  # 13
    data_list = list()
    for idx, item in enumerate(np.array(df).tolist()):
        category = item[2]
        content = item[4]
        title = item[8]
        label = item[9]
        if category is np.nan or content is np.nan or title is np.nan or label is np.nan:
            print("error in ", idx)
            continue
        if label not in ["适合", "不适合"]:
            print("error: ", label)
            continue
        new_line = "__label__" + label + "\t" + " ".join([_ for _ in category]) + " " + " ".join(
            [_ for _ in title]) + " " + " ".join([_ for _ in content])
        data_list.append(new_line)

    raw_train_data = list()
    with open(raw_data, encoding="utf8") as f:
        for line in f:
            raw_train_data.append(line.strip())
    print("len of raw train data: %d" % len(raw_train_data))
    data_list.extend(raw_train_data)
    print("len of new expanded train data: %d" % len(data_list))
    with open(output_data, "w", encoding="utf8") as fw:
        for idx, line in enumerate(data_list):
            fw.write(line + "\n")


def bert_classifier_helper(data_path, output_path, mode='test'):
    with open(output_path, "w", encoding="utf8") as fw:
        with open(data_path, encoding="utf8") as f:
            for idx, line in enumerate(f):
                tokens = line.strip().split("\t")
                assert len(tokens) == 11
                title = tokens[8]
                content = tokens[4]
                label = tokens[9]
                if label not in ["适合", "不适合"]:
                    continue
                guid = mode + "-" + str(idx)
                json_str = json.dumps(
                    {"id": guid, "title": title, "content": content.replace("\\r\\n", "[PAR]"), "label": label},
                    ensure_ascii=False)
                fw.write(json_str + "\n")


def bert_classifier_helper2(excel_data, output_path, mode='train'):
    df = pd.read_excel(excel_data)  # 13
    with open(output_path, "w", encoding="utf8") as fw:
        for idx, item in enumerate(np.array(df).tolist()):
            content = item[4]
            title = item[8]
            label = item[9]
            if label not in ["适合", "不适合"]:
                continue
            guid = mode + "-" + str(idx + 12455)
            json_str = json.dumps(
                {"id": guid, "title": title, "content": content.replace("\\r\\n", "[PAR]"), "label": label},
                ensure_ascii=False)
            fw.write(json_str + "\n")


if __name__ == '__main__':
    excel_data_path = "/Users/geofftong/PycharmProjects/car_news_sum/data_dir/news/classifier/p2.news.classfier.xlsx"
    output_data_prefix = "/Users/geofftong/PycharmProjects/car_news_sum/data_dir/news/classifier/news.classfier"
    # convert_excel_data(excel_data_path, output_data_prefix)
    # convert_excel_data2(excel_data="../data_dir/news/classifier/p2.news.classfier.xlsx",
    #                     raw_data="../data_dir/news/classifier/news.classfier.train",
    #                     output_data="../data_dir/news/classifier/news.classfier.train2")
    # bert_classifier_helper(data_path="../data_dir/news/classifier/news.classfier.test.raw",
    #                        output_path="../data_dir/news/classifier/news.bert.test")
    bert_classifier_helper2(excel_data=excel_data_path,
                            output_path="../data_dir/news/classifier/news.bert.train2")
