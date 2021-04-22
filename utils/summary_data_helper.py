#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: geofftong
# Mail: geofftong@tencent.com
# Created Time: 2019/6/26 3:13 PM
import copy
import json
import random
from collections import OrderedDict

import numpy as np
import pandas as pd
import requests

random.seed(6)  # before 11.25: 16 11.28: 116


# todo: norm chinese punc

def category_helper(data_path):
    cate_dict = dict()
    with open(data_path, encoding="utf8") as f:
        for idx, line in enumerate(f):
            news_item = json.loads(line.strip())
            category = news_item["category"]
            if category not in cate_dict:
                cate_dict[category] = 1
            else:
                cate_dict[category] += 1
    for cate, cnt in sorted(cate_dict.items(), key=lambda d: d[1], reverse=True):
        print(cate, cnt)


def ch_sent_split(content, high_recall=True):
    """
    中文断句，remains punctuation
    :param content:
    :param high_recall: 允许最后一句不以标点符号结尾
    :return:
    """
    # content = content.replace(";", "；").replace("!", "！").replace("?", "？")
    all_sentence = []
    sent_cut_punc = ["。", "！", "？", "!", "?"]  # todo: "；", ";"
    special_punc = ["’", "”", ")", "）", "]", ">", "》", "}", "」", "】"]
    w_idx, s_idx = 0, 0
    current_sentence = ""
    while w_idx < len(content):
        word = content[w_idx]
        if word not in sent_cut_punc:
            current_sentence += word
        else:
            current_sentence += word
            while w_idx + 1 < len(content) and content[w_idx + 1] in (special_punc + sent_cut_punc):
                current_sentence += content[w_idx + 1]
                w_idx += 1
            if len(copy.copy(current_sentence)) > 0:  # > 1
                all_sentence.append(current_sentence)
            s_idx = w_idx
            current_sentence = ""
        w_idx += 1
    s_idx += 1 if s_idx > 0 else 0  # s_idx=0
    if high_recall and s_idx != w_idx:
        all_sentence.append(content[s_idx:])
    return all_sentence


def json_to_excel(input, output):
    data = []  # 用于存储每一行的Json数据
    with open(input, 'r', encoding='UTF-8') as fr:
        for idx, line in enumerate(fr):
            j = json.loads(line, object_pairs_hook=OrderedDict)
            data.append(j)  # [{}]

    df = pd.DataFrame()  # 最后转换得到的结果
    for line in data:
        df1 = pd.DataFrame([line])
        df = df.append(df1)

    # 在excel表格的第1列写入, 不写入index
    df.to_excel(output, sheet_name='Data', startcol=0, index=False)


def norm_news_data(raw_data_path, output_data_path):
    """
    过滤无用的新闻类别，统计新闻平均长度
    :param raw_data_path:
    :param output_data_path:
    :return:
    """
    samples_dict = dict()
    samples_cnt = dict()
    unused_category = ["旅游", "时尚"]
    unused_sub_category = ["美食_菜谱", "汽车_汽车评测"]
    unused_keywords = ["财经24小时", "新闻汇总", "新闻汇编", "寻人启事", "征集违法犯罪线索", "招生说明"]  # 通告 通知
    with open(raw_data_path, encoding="utf8") as f:
        for idx, line in enumerate(f):
            valid_flag = True
            news_dict = json.loads(line.strip())
            title = news_dict["title"]
            vSubject = news_dict["vSubject"]
            if len(vSubject) == 0:
                continue
            category = vSubject[0]["sName"]
            if category in unused_category:
                valid_flag = False
            for u_cat in unused_sub_category:
                for item in vSubject:
                    if u_cat in item["sName"]:
                        valid_flag = False
            for u_words in unused_keywords:
                if u_words in title:
                    valid_flag = False
            if len(news_dict["content"].replace("\\r\\n", "")) <= 350:
                valid_flag = False
            if valid_flag:
                if category not in samples_dict:
                    samples_dict[category] = [news_dict]
                    samples_cnt[category] = 1
                else:
                    samples_dict[category].append(news_dict)
                    samples_cnt[category] += 1
    # print(len(samples_dict))
    demo_data = list()
    for cate, cnt in sorted(samples_cnt.items(), key=lambda d: d[1], reverse=True):
        # print(str(cnt) + "\t" + cate)
        demo_data.extend(samples_dict[cate])  # [:3]
    random.shuffle(demo_data)
    # id doc_id title category url content(sentences list) label
    with open(output_data_path, "w", encoding="utf8") as fw:
        for idx, item in enumerate(demo_data):
            segmented_content = list()
            for para in item["content"].split("\\r\\n"):
                if len(para.strip()) != 0:
                    segmented_content.append("[SEP]".join(ch_sent_split(para.strip())))
            temp_dict = {"idx": idx + 10151, "docId": item["docId"], "title": item["title"],
                         "category": item["vSubject"][0]["sName"], "url": item["url"],
                         "content": "[PAR]".join(segmented_content),
                         "result": ""}
            fw.write(json.dumps(temp_dict, ensure_ascii=False) + "\n")


def gen_label(ext_sents, sent_list):
    punc_list = ["", "。", "!", "！", "?", "？"]  # ";", "；"
    label_list = [0 for _ in sent_list]
    for item in ext_sents:
        flag = False
        for punc in punc_list:
            if item + punc in sent_list:
                flag = True
                label_list[sent_list.index(item + punc)] = 1
                break
            if item[:-1] + punc in sent_list:
                flag = True
                label_list[sent_list.index(item[:-1] + punc)] = 1
                break
            for raw_sent in sent_list:
                if item in raw_sent or item[:-1] in raw_sent:  # or edit_distance(item, raw_sent) < 4
                    flag = True
                    label_list[sent_list.index(raw_sent)] = 1
                    break
        if not flag:
            print("errors in ", item)
    if len(ext_sents) != sum(label_list):
        print("errors in: ", ext_sents, sent_list)
        print("label: ", len(ext_sents), sum(label_list))
    return label_list


def labeled_excel_to_list(excel_data, output):
    df = pd.read_excel(excel_data)  # usecols=[1, 2, 3, 4, 5, 6, 7]
    data = np.array(df)
    match_error_list = list()
    temp_list = list()

    with open("format.error", "w", encoding="utf8") as fwe:
        with open(output, "w", encoding="utf8") as fw:
            for item in data.tolist():
                valid_flag = True
                # idx, doc_id, title, category, url, content, result = item
                title, doc_id, _, url, result, content, bold = item
                if result is np.nan:
                    print("nan error:", doc_id)
                    continue
                segmented_content = [sent.strip() for sent in content.strip().replace("[PAR]", "[SEP]").split("[SEP]")]
                segmented_result = [sent.strip() for sent in ch_sent_split(str(result.strip()))]
                for sent in segmented_result:
                    if sent.strip() not in segmented_content:
                        print("match error:", doc_id, sent.strip(), segmented_content)
                        fwe.write(str(
                            doc_id) + "\t" + title.strip() + "\n" + "selected_sentence: " + sent.strip() + "\n" + "content: " + content + "\n\n")
                        temp_list.append({"id": doc_id, "selected_sentence": sent.strip(), "title": title.strip(),
                                          "url": url.strip(), "anno_result": result, "content": content})

                        valid_flag = False
                        match_error_list.append(doc_id)
                        break
                if valid_flag:
                    extract_label = gen_label(segmented_result, segmented_content)
                    valid_labeled_dict = {"id": doc_id, "title": title.strip(), "url": url.strip(),
                                          "extract_label": extract_label, "extract_sentence": segmented_result,
                                          "content": content.strip(), "bold": str(bold).strip()}
                    json_str = json.dumps(valid_labeled_dict, ensure_ascii=False)
                    fw.write(json_str + "\n")

    df = pd.DataFrame()  # 最后转换得到的结果
    for line in temp_list:
        df1 = pd.DataFrame([line])
        df = df.append(df1)

    # 在excel表格的第1列写入, 不写入index
    df.to_excel("format.error.xlsx", sheet_name='Data', startcol=0, index=False)

    print(len(match_error_list), match_error_list)  # 1936 64   --> 1287


def generate_data_to_be_labeled(all_news, output_data):
    """
    generating data by category, start_id=4000
    :return:
    """
    start_id = 4000
    cate_dict = dict()
    gen_list = list()
    uniqle_docId = set()  # 去重
    with open(all_news, encoding="utf8") as f:
        for idx, line in enumerate(f):
            news_item = json.loads(line.strip())
            if news_item["idx"] < start_id:
                continue
            if news_item["docId"] in uniqle_docId:
                continue
            else:
                uniqle_docId.add(news_item["docId"])
            if news_item["category"] not in cate_dict:
                cate_dict[news_item["category"]] = [news_item]
            else:
                cate_dict[news_item["category"]].append(news_item)
    for cate in cate_dict:
        # print(cate, len(cate_dict[cate]))
        if len(cate_dict[cate]) < 10:
            continue
        elif len(cate_dict[cate]) < 500:
            gen_list.extend(cate_dict[cate])
        else:
            if cate == "汽车":
                gen_list.extend(random.sample(cate_dict[cate], 453))
            else:
                gen_list.extend(random.sample(cate_dict[cate], 500))
    uniqle_nid = set()
    with open(output_data, "w", encoding="utf8") as fw:
        for idx, item in enumerate(gen_list):
            uniqle_nid.add(item["docId"])
            fw.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(len(uniqle_nid))


def review_labeled_data(json_data_path, output_path):
    news_item_list = list()
    with open(json_data_path, encoding="utf8") as f:
        result_list = json.load(f)
    for item in result_list:
        badcase_flag = False
        answer_label = list()
        for user_label_dict in item["answer"]:  # answers
            label_dict = eval(user_label_dict["selects"].replace("true", "True").replace("false", "False"))  # label
            if label_dict["is_badcase"] is True:
                badcase_flag = True
                break
            answer_label = label_dict["items"]
        if badcase_flag:
            continue
        content_list = item["content_data"]["content"]
        assert len(answer_label) == len(content_list)

        answer_text = ""
        invalid_label_flag = False
        for pos, la in enumerate(answer_label):
            if la == "":
                invalid_label_flag = True
            if la == "1":
                answer_text += content_list[pos]
        if invalid_label_flag:
            continue
        news_item_list.append(
            OrderedDict(
                idx=item["content_data"]["idx"], docId=item["content_data"]["docId"], title=item["content"],
                category=item["content_data"]["category"], url=item["content_data"]["url"],
                content="[SEP]".join(content_list), labeled_result=answer_text))
    random.shuffle(news_item_list)
    print(len(news_item_list))

    df = pd.DataFrame()  # 最后转换得到的结果
    for line in news_item_list[:100]:
        df1 = pd.DataFrame([line])
        df = df.append(df1)

    # 在excel表格的第1列写入, 不写入index
    df.to_excel(output_path, sheet_name='Data', startcol=0, index=False)


def generate_data_to_be_labeled_by_category(all_news, output_data):
    """
    generating data by category, start_id=4000
    :return:
    """
    start_id = 4000
    cate_dict = dict()
    gen_list = list()
    uniqle_docId = set()  # 去重
    with open(all_news, encoding="utf8") as f:
        for idx, line in enumerate(f):
            news_item = json.loads(line.strip())
            if news_item["idx"] < start_id:
                continue
            if news_item["docId"] in uniqle_docId:
                continue
            else:
                uniqle_docId.add(news_item["docId"])
            if news_item["category"] not in cate_dict:
                cate_dict[news_item["category"]] = [news_item]
            else:
                cate_dict[news_item["category"]].append(news_item)
    for cate in cate_dict:
        # print(cate, len(cate_dict[cate]))
        if len(cate_dict[cate]) < 50:
            continue
        print(cate)
        gen_list.extend(random.sample(cate_dict[cate], 50))

    uniqle_nid = set()
    with open(output_data, "w", encoding="utf8") as fw:
        for idx, item in enumerate(gen_list):
            uniqle_nid.add(item["docId"])
            fw.write(
                json.dumps({"id": item["idx"], "title": item["title"], "url": item["url"], "category": item["category"],
                            "extract_label": [0 for _ in
                                              item["content"].strip().replace("[PAR]", "[SEP]").split("[SEP]")],
                            "content": item["content"].strip()}, ensure_ascii=False) + "\n")
    print(len(uniqle_nid))


def format_data_to_be_trained(json_data_path, raw_data, output_path, max_len=3200, max_labels_size=75):
    """ process labeled data set"""
    url_to_content = dict()
    mismatch_cnt, match_cnt = 0, 0
    with open(raw_data, encoding="utf8") as f:
        for line in f:
            temp_dict = json.loads(line.strip())
            url_to_content[temp_dict["url"].strip()] = temp_dict["content"]

    with open(json_data_path, encoding="utf8") as f:
        result_list = json.load(f)
    with open(output_path, "w", encoding="utf8") as fw:
        for item in result_list:
            badcase_flag = False
            answer_label = list()
            for user_label_dict in item["answer"]:  # answers
                label_dict = eval(user_label_dict["selects"].replace("true", "True").replace("false", "False"))  # label
                if label_dict["is_badcase"] is True:
                    badcase_flag = True
                    break
                answer_label = label_dict["items"]
            if badcase_flag:
                continue
            # content_list = item["content_data"]["content"]
            if item["content_data"]["url"].strip() not in url_to_content:
                mismatch_cnt += 1
                continue

            content_list = url_to_content[item["content_data"]["url"].strip()].replace("[PAR]", "[SEP]").split("[SEP]")
            assert len(answer_label) == len(content_list)

            answer_text = list()
            invalid_label_flag = False
            for pos, la in enumerate(answer_label):
                if la == "":
                    invalid_label_flag = True
                if la == "1":
                    answer_text.append(content_list[pos])
            if invalid_label_flag:
                continue

            if len(url_to_content[item["content_data"]["url"].strip()].replace("[SEP]", "").replace("[PAR]",
                                                                                                    "")) > max_len:
                continue
            if len(answer_label) > max_labels_size:
                continue

            json_str = json.dumps(
                {"id": item["content_data"]["idx"], "title": item["content"], "url": item["content_data"]["url"],
                 "extract_label": answer_label, "extract_sentence": answer_text,
                 "category": item["content_data"]["category"],
                 "content": url_to_content[item["content_data"]["url"].strip()]}, ensure_ascii=False)
            fw.write(json_str + "\n")
            match_cnt += 1
    print(mismatch_cnt, match_cnt)


def get_all_hits(mission_id, output):
    """ get labeled data from wx_test platform"""
    data = {
        "mission_id": mission_id,
        "status": "all",
        "page_size": 40000
    }
    url = "http://mmtest.oa.com/crowdsourcing_admin/api/verify-answer"
    resp = requests.get(url, params=data)
    res = resp.json()
    with open(output, "w", encoding="utf8") as fw:
        json_str = json.dumps(res['data']['hits'], ensure_ascii=False)
        fw.write(json_str + "\n")
    return res['data']['hits']


if __name__ == '__main__':
    # filter unused news category
    # norm_news_data(raw_data_path="/Users/geofftong/PycharmProjects/car_news_sum/data_dir/news/raw_data/online/news.para.0715-0721",
    #                output_data_path="/Users/geofftong/PycharmProjects/car_news_sum/data_dir/news/raw_data/news.para.0715-0721.json")

    # sampling news by used category
    # category_helper(data_path="/Users/geofftong/PycharmProjects/car_news_sum/data_dir/news/raw_data/news.para.0715-0721.json")

    # convert json data to excel
    # json_to_excel("data_dir/news.para.1w.json", "data_dir/news.all_v2.xls")

    # convert json data to excel
    # labeled_excel_to_list(
    #     excel_data="/Users/geofftong/PycharmProjects/car_news_sum/data_dir/news/raw_data/excel/labeled]news.p1.2000_bold.xlsx",
    #     output="p2.2000_2.json")

    # generate_data_to_be_labeled(
    #     all_news="/Users/geofftong/PycharmProjects/car_news_sum/data_dir/news/raw_data/news.para.json",
    #     output_data="/Users/geofftong/PycharmProjects/car_news_sum/data_dir/news/raw_data/news.p1_2.5000.json")

    # generate_data_to_be_labeled_by_category(
    #     all_news="/Users/geofftong/PycharmProjects/car_news_sum/data_dir/news/raw_data/news.para.json",
    #     output_data="/Users/geofftong/Pyc1harmProjects/car_news_sum/data_dir/news/raw_data/category.test.json")

    # pull data from online platform
    get_all_hits(mission_id="$oid_5dc53a48aafd71561fbf8fed",
                 output="/Users/geofftong/PycharmProjects/car_news_sum/data_dir/news/summary/新闻摘要第三期1210.txt")

    # sample labeled data to be checked up
    review_labeled_data(
        json_data_path="/Users/geofftong/PycharmProjects/car_news_sum/data_dir/news/summary/新闻摘要第三期1210.txt",
        output_path="/Users/geofftong/PycharmProjects/car_news_sum/data_dir/news/summary/新闻摘要第三期抽查100_v4.xlsx")

    # format valid labeled data to be trained
    # format_data_to_be_trained(
    #     json_data_path="/Users/geofftong/PycharmProjects/car_news_sum/data_dir/news/summary/新闻摘要第二期1108.txt",
    #     raw_data="/Users/geofftong/PycharmProjects/car_news_sum/data_dir/news/raw_data/news.p1_2.5000.json",
    #     output_path="/Users/geofftong/PycharmProjects/car_news_sum/data_dir/news/summary/p2.2035.json")
