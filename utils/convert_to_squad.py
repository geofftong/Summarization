#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: geofftong
# Mail: geofftong@tencent.com
# Created Time: 2019/2/27 6:18 PM`
import json
import copy
from utils import ngram_overlap


def norm_punc_on_bert(text):
    return text.replace('“', '\"').replace('”', '\"')


def statistics(filename):
    sent_all_num, over_num = 0, 0
    max_sent_len, max_cont_len, max_word_len = 0, 0, 0
    with open(filename, encoding="utf8") as f:
        for idx, line in enumerate(f):
            sample = json.loads(line.strip())
            sent_len = len(sample["extract_label"].replace("[SEP]", "[PAR]").split("[PAR]"))
            sent_list = sample["content"].replace("[SEP]", "[PAR]").split("[PAR]")
            for item in sent_list:
                sent_all_num += 1
                word_len = len(item)
                max_word_len = max(word_len, max_word_len)
                if word_len > 72:
                    over_num += 1
                    print("****", sample["id"], word_len, item)
            cont_len = len(sample["content"])
            max_sent_len = max(max_sent_len, sent_len)
            max_cont_len = max(max_cont_len, cont_len)

        print(max_sent_len, max_word_len, max_cont_len)
        print(sent_all_num, over_num, float(over_num * 1.0/sent_all_num))


def search_answer_span(text, content):
    """

    :param text:
    :param content:
    :return:
    """
    idx = content.find(text)  # index
    if idx == -1:
        print(text + "|||" + content)
        print("mismatch")
    return idx


def convert_squad_from_baike(raw_data):
    """

    :param raw_data:
    :return:
    """
    mr_data = dict()
    mr_data["version"] = "v2.0"
    mr_data["data"] = list()

    for _, qa_pair in enumerate(raw_data):
        samples = dict()
        samples["title"] = ""

        paragraph = dict()
        paragraph["qas"] = list()
        qas = dict()
        normed_context = norm_punc_on_bert(qa_pair["content"])
        paragraph["context"] = normed_context.replace("[SEP]", "").replace("[PAR]", "")
        paragraph["segmented_context"] = normed_context.replace("[PAR]", "[SEP]").split("[SEP]")
        qas["question"] = norm_punc_on_bert(qa_pair["title"].replace("_", ""))  # _
        qas["id"] = str(qa_pair["id"])  # string

        if "extract_label" not in qa_pair:  # for prediction
            qa_pair["extract_label"] = ["0" for _ in paragraph["segmented_context"]]
        if "extract_state" not in qa_pair:
            qa_pair["extract_state"] = ["0" for _ in paragraph["segmented_context"]]
        extract_label = qa_pair["extract_label"]  # replace("[PAR]", "[SEP]").split("[SEP]")

        assert len(paragraph["segmented_context"]) == len(extract_label)
        qas_answer = dict()
        if "1" in qa_pair["extract_label"]:
            first_pos = extract_label.index("1")
            qas_answer["text"] = paragraph["segmented_context"][first_pos]
            qas["is_impossible"] = False
        else:
            qas_answer["text"] = ""
            qas["is_impossible"] = True
        qas_answer["answer_start"] = search_answer_span(qas_answer["text"], paragraph["context"])
        if qas_answer["answer_start"] == -1:
            continue
        qas["answers"] = [qas_answer]
        qas["extract_state"] = qa_pair["extract_state"]

        # 1018
        if "repeat_3_gram" in qa_pair:
            qas["repeat_3_gram"] = qa_pair["repeat_3_gram"]
        if "repeat_4_gram" in qa_pair:
            qas["repeat_4_gram"] = qa_pair["repeat_4_gram"]
        if "repeat_5_gram" in qa_pair:
            qas["repeat_5_gram"] = qa_pair["repeat_5_gram"]
        if "repeat_6_gram" in qa_pair:
            qas["repeat_6_gram"] = qa_pair["repeat_6_gram"]
        if "repeat_7_gram" in qa_pair:
            qas["repeat_7_gram"] = qa_pair["repeat_7_gram"]

        paragraph["qas"] = [qas]
        samples["paragraphs"] = [paragraph]
        mr_data["data"].append(samples)
    return mr_data


def expand_history_data(json_list, max_mem=1, max_hist_sent=20, data_type='train'):
    """
    todo: text_b mask; no order?
    :param json_list: id title content extract_label
    :param data_type:
    :param max_mem: maximum number history sentence
    :return:
    """
    expand_list = list()
    for raw_sample in json_list:
        sample_id = 0
        title = raw_sample["title"].replace("_", "")
        content = raw_sample["content"].replace("[PAR]", "[SEP]").split("[SEP]")
        extract_label = [str(l) for l in raw_sample["extract_label"]]  # replace("[PAR]", "[SEP]").split("[SEP]")
        assert len(content) == len(extract_label)

        turn_idx = 0
        extract_state = [0 for _ in range(len(extract_label))]

        exist_gold_sentence = True
        selected_sentecnces = list()
        while exist_gold_sentence:
            turn_idx += 1
            expand_sample = dict()
            expand_sample["id"] = data_type + "-" + str(raw_sample["id"]) + "-" + str(sample_id)
            if len(selected_sentecnces) == 0:  # todo: 0904
                expand_sample["title"] = title
            else:
                # history = [sent[:max_hist_sent] for sent in selected_sentecnces[-max_mem:]]
                history = [sent for sent in selected_sentecnces[-max_mem:]]
                expand_sample["title"] = title + "[SEP]" + "[SEP]".join(history)
            expand_sample["content"] = raw_sample["content"]  # 0430: content
            expand_sample["extract_label"] = copy.copy(extract_label)

            # sentence-level feature
            expand_sample["extract_state"] = copy.copy(extract_state)

            # 1018: word-level feature
            expand_sample["repeat_3_gram"] = ngram_overlap("".join(content), "".join(selected_sentecnces), 3)
            expand_sample["repeat_4_gram"] = ngram_overlap("".join(content), "".join(selected_sentecnces), 4)
            expand_sample["repeat_5_gram"] = ngram_overlap("".join(content), "".join(selected_sentecnces), 5)
            expand_sample["repeat_6_gram"] = ngram_overlap("".join(content), "".join(selected_sentecnces), 6)
            expand_sample["repeat_7_gram"] = ngram_overlap("".join(content), "".join(selected_sentecnces), 7)

            expand_list.append(expand_sample)
            sample_id += 1
            if "1" in extract_label:
                first_pos = extract_label.index("1")
                selected_sentecnces.append(content[first_pos])
                extract_label[first_pos] = "0"
                extract_state[first_pos] = turn_idx
                expand_sample["is_impossible"] = False

            else:
                expand_sample["is_impossible"] = True
                exist_gold_sentence = False
    # print(len(json_list), len(expand_list))
    return expand_list


def convert_span_squad_from_baike(raw_data):
    """
    norm punc
    :param raw_data:
    :return:
    """
    mr_data = dict()
    mr_data["version"] = "v2.0"
    mr_data["data"] = list()

    for _, qa_pair in enumerate(raw_data):
        samples = dict()
        samples["title"] = ""

        paragraph = dict()
        paragraph["qas"] = list()
        qas = dict()
        paragraph["extract_state"] = qa_pair["extract_state"]
        normed_context = norm_punc_on_bert(qa_pair["content"])
        paragraph["context"] = normed_context.replace("[SEP]", "").replace("[PAR]", "")
        paragraph["segmented_context"] = normed_context.replace("[PAR]", "[SEP]").split("[SEP]")

        qas["question"] = norm_punc_on_bert(qa_pair["title"].replace("_", ""))  # _
        qas["id"] = str(qa_pair["id"])  # string

        if "extract_label" not in qa_pair:  # for prediction
            qa_pair["extract_label"] = ["0" for _ in paragraph["segmented_context"]]

        extract_label = qa_pair["extract_label"]  # replace("[PAR]", "[SEP]").split("[SEP]")

        assert len(paragraph["segmented_context"]) == len(extract_label)
        qas_answer = dict()
        qas_answer["text"] = ""
        if "1" in qa_pair["extract_label"]:
            first_pos = extract_label.index("1")
            while extract_label[first_pos] == "1":  # 连续span
                qas_answer["text"] += paragraph["segmented_context"][first_pos]
                if first_pos == len(extract_label) - 1:
                    break
                first_pos += 1
            qas["is_impossible"] = False
        else:
            qas["is_impossible"] = True
        qas_answer["answer_start"] = search_answer_span(qas_answer["text"], paragraph["context"])
        if qas_answer["answer_start"] == -1:
            print("warning: could not find answer from %s and %s" %(qas_answer["text"], paragraph["context"]))
            continue
        qas["answers"] = [qas_answer]

        paragraph["qas"] = [qas]
        samples["paragraphs"] = [paragraph]
        mr_data["data"].append(samples)
    return mr_data


def expand_span_history_data(json_list, max_mem=1, data_type='train'):
    """
    todo: text_b mask; no order?
    :param json_list: id title content extract_label
    :param data_type:
    :param max_mem: maximum number history sentence
    :return:
    """
    expand_list = list()
    for raw_sample in json_list:
        sample_id = 0
        title = raw_sample["title"].replace("_", "")
        content = raw_sample["content"].replace("[PAR]", "[SEP]").split("[SEP]")
        extract_label = [str(l) for l in raw_sample["extract_label"]]  # replace("[PAR]", "[SEP]").split("[SEP]")
        assert len(content) == len(extract_label)
        turn_idx = 0
        extract_state = [0 for _ in range(len(extract_label))]

        exist_gold_sentence = True
        selected_sentecnces = list()
        while exist_gold_sentence:
            turn_idx += 1
            expand_sample = dict()
            expand_sample["id"] = data_type + "-" + str(raw_sample["id"]) + "-" + str(sample_id)
            if len(selected_sentecnces) == 0:
                expand_sample["title"] = title
            else:
                # history = [sent[:max_hist_sent] for sent in selected_sentecnces[-max_mem:]]
                history = [sent for sent in selected_sentecnces[-max_mem:]]
                expand_sample["title"] = title + "[SEP]" + "[SEP]".join(history)
            expand_sample["content"] = raw_sample["content"]  # 0430: content
            expand_sample["extract_label"] = copy.copy(extract_label)
            expand_sample["extract_state"] = copy.copy(extract_state)
            expand_list.append(expand_sample)
            sample_id += 1
            if "1" in extract_label:
                first_pos = extract_label.index("1")
                while extract_label[first_pos] == "1":  # 连续span
                    selected_sentecnces.append(content[first_pos])
                    extract_label[first_pos] = "0"
                    extract_state[first_pos] = turn_idx
                    if first_pos == len(extract_label) - 1:
                        break
                    first_pos += 1
                expand_sample["is_impossible"] = False
            else:
                expand_sample["is_impossible"] = True
                exist_gold_sentence = False
    # print(len(json_list), len(expand_list))
    return expand_list


def beam_search_negative_sampling(json_list, max_mem=1, data_type='train'):
    """
    extract_state: history state
    :param json_list:
    :param max_mem:
    :param data_type:
    :return:
    """
    expand_list = list()
    for raw_sample in json_list:
        sample_id = 0
        title = raw_sample["title"].replace("_", "")
        content = raw_sample["content"].replace("[PAR]", "[SEP]").split("[SEP]")
        extract_label = [str(l) for l in raw_sample["extract_label"]]  # replace("[PAR]", "[SEP]").split("[SEP]")
        assert len(content) == len(extract_label)
        turn_idx = 0
        selected_sentences, fake_selected_sentences = list(), list()
        extract_state, fake_extract_state = [0 for _ in range(len(extract_label))], [0 for _ in range(len(extract_label))]
        exist_neg_flag = False
        for sent_idx in range(len(extract_label)):
            if extract_label[sent_idx] == "1":
                turn_idx += 1

                # add a positive example
                pos_sample = dict()
                pos_sample["id"] = data_type + "-" + str(raw_sample["id"]) + "-" + str(sample_id)
                pos_sample["content"] = raw_sample["content"]  # 0430: content
                if len(selected_sentences) == 0:
                    pos_sample["title"] = title
                else:
                    history = [sent for sent in selected_sentences[-max_mem:]]
                    pos_sample["title"] = title + "[SEP]" + "[SEP]".join(history)
                pos_sample["is_impossible"] = False
                pos_sample["extract_label"] = copy.copy(extract_label)
                pos_sample["extract_state"] = copy.copy(extract_state)
                expand_list.append(pos_sample)
                sample_id += 1

                # todo：add more than one "fake" no-answer-exam from 2rd turn (select wrong sentence in history summary)
                if exist_neg_flag:
                    neg_sample = dict()
                    neg_sample["id"] = data_type + "-" + str(raw_sample["id"]) + "-" + str(sample_id)
                    neg_sample["content"] = raw_sample["content"]

                    history = [sent for sent in fake_selected_sentences[-max_mem:]]
                    neg_sample["title"] = title + "[SEP]" + "[SEP]".join(history)

                    neg_sample["is_impossible"] = True
                    neg_sample["extract_label"] = ["0" for _ in range(len(extract_label))]
                    neg_sample["extract_state"] = copy.copy(fake_extract_state)
                    expand_list.append(neg_sample)
                    sample_id += 1

                # update state for next turn
                fake_extract_state = copy.copy(extract_state)
                fake_selected_sentences = copy.copy(selected_sentences)
                selected_sentences.append(content[sent_idx])
                extract_state[sent_idx] = turn_idx
                extract_label[sent_idx] = "0"

                # update fake state for next turn
                if sent_idx + 1 < len(extract_label) - 1 and extract_label[sent_idx + 1] == "0":
                    exist_neg_flag = True
                    fake_extract_state[sent_idx + 1] = turn_idx
                    fake_selected_sentences.append(content[sent_idx + 1])
                else:
                    exist_neg_flag = False
            # add a no-answer example
            if sent_idx == len(extract_label) - 1:
                neg_sample = dict()
                neg_sample["id"] = data_type + "-" + str(raw_sample["id"]) + "-" + str(sample_id)
                neg_sample["content"] = raw_sample["content"]
                if len(selected_sentences) == 0:
                    neg_sample["title"] = title
                else:
                    history = [sent for sent in selected_sentences[-max_mem:]]
                    neg_sample["title"] = title + "[SEP]" + "[SEP]".join(history)
                neg_sample["is_impossible"] = True
                neg_sample["extract_label"] = ["0" for _ in range(len(extract_label))]
                neg_sample["extract_state"] = copy.copy(extract_state)
                expand_list.append(neg_sample)
    # print(len(json_list), len(expand_list))
    return expand_list


if __name__ == '__main__':
    train_flag = False      # test data does not need to expand
    neg_sampling_flag = False
    file_path = "../data_dir/news/summary/第二期/temp.test"  # p1.news.span.train
    expand_file_path = "../data_dir/news/summary/第二期/news_ch_5235.train.expand"  # news.train.2875.expand
    squad_file = "../data_dir/news/summary/第二期/temp.test.squad"  # news.train.2875.expand.squad

    raw_data_list = list()
    with open(file_path, encoding="utf8") as f:
        for idx, line in enumerate(f):
            raw_data_list.append(json.loads(line.strip()))

    if train_flag:
        if neg_sampling_flag:
            expand_data = beam_search_negative_sampling(raw_data_list)
        else:
            expand_data = expand_history_data(raw_data_list)
        with open(expand_file_path, "w", encoding="utf8") as fw:
            for idx, line in enumerate(expand_data):
                json_str = json.dumps(line, ensure_ascii=False)
                fw.write(json_str + "\n")  # geoff: diff from squad
        squad_data = convert_squad_from_baike(expand_data)
    else:
        squad_data = convert_squad_from_baike(raw_data_list)

    with open(squad_file, "w", encoding="utf8") as fw:
        json_str = json.dumps(squad_data, ensure_ascii=False)
        fw.write(json_str + "\n")
    print(len(raw_data_list), len(squad_data["data"]))   # 2875 15641
