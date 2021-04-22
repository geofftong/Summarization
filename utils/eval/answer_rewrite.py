#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: geofftong
# Mail: geofftong@tencent.com
# Created Time: 2019/5/2 12:59 PM
import codecs
import json


# todo: 指代问题
def answer_rewrite(text, raw_text):
    text = text.replace(",", "，")  # todo: 而
    invalid_start_words = ["此外，", "因此，", "所以，", "并且，", "再者，", "另外，", "另一方面，", "而且，", "于是，", "然而，", "不过，", "但是，",
                           "总之，", "之后，", "终于，", "后来，", "那么，", "但，",
                           "此外", "因此", "所以", "并且", "再者", "另外", "另一方面", "而且", "于是", "然而", "不过", "但是",
                           "总之", "之后", "终于", "后来", "那么", "但", "又", "。", "，", "！", "？", "；", ".", ",", "!", "?", ";"]
    if len(raw_text.replace("[PAR]", "").replace("[SEP]", "")) < 30:
        text = raw_text.replace("[PAR]", "[SEP]")
    for invalid_word in invalid_start_words:
        if text.startswith(invalid_word):
            text = text[len(invalid_word):]
            break
    if text.endswith("；") or text.endswith(";"):
        text = text[:-1] + "。"
    return text


uniq_id = list()
dataset = list()
with codecs.open("xiaowei.online.8w", encoding="utf8") as f:
    for idx, line in enumerate(f):
        item = json.loads(line.strip())
        if item["id"] in uniq_id:
            print("error in ", item["id"])
            continue
        # assert item["id"] not in uniq_id
        uniq_id.append(item["id"])
        dataset.append(item)

# with codecs.open("../data_dir/preprocessed/all.test.json", encoding="utf8") as f:
#     for idx, line in enumerate(f):
#         item = json.loads(line.strip())
#         assert item["id"] not in uniq_id
#         uniq_id.append(item["id"])
#         dataset.append(item)
#
# with codecs.open("baike.online.1", encoding="utf8") as f:
#     for idx, line in enumerate(f):
#         item = json.loads(line.strip())
#         assert item["id"] not in uniq_id
#         uniq_id.append(item["id"])
#         dataset.append(item)
#
# with codecs.open("baike.online.2", encoding="utf8") as f:
#     for idx, line in enumerate(f):
#         item = json.loads(line.strip())
#         assert item["id"] not in uniq_id
#         uniq_id.append(item["id"])
#         dataset.append(item)

print(len(uniq_id), len(dataset))

# test
rewrite_len, content_len = 0, 0
with codecs.open("online.display2", "w", encoding="utf8") as fw1:
    for item in dataset:
        assert len(item["extract_sentence"]) > 0

        extract_sentence = "[SEP]".join(item["extract_sentence"])
        content = item["content"]
        corrected_sentence = answer_rewrite(extract_sentence, content)

        rewrite_len += len(corrected_sentence.replace("[SEP]", ""))
        content_len += len(content.replace("[SEP]", ""))
        print(corrected_sentence.replace("[SEP]", ""))
        fw1.write(str(item["id"]) + ". " + item["title"] + "\n" + "raw answer: " + content.replace("[SEP]",
                                                                                                   "") + "\n" + "rewrite answer: " + corrected_sentence.replace(
            "[SEP]", "") + "\n\n")
        # if extract_sentence != corrected_sentence:
        #     print(item["id"], item["title"], extract_sentence, "#####", corrected_sentence)

        # "rewrite_sentence": item["rewrite_sentence"], category
        json_str = {"id": item["id"], "title": item["title"],
                    "extract_sentence": corrected_sentence.split("[SEP]"),
                    "extract_label": item["extract_label"], "content": item["content"]}
print(len(dataset))
print("rewrite_len: ", rewrite_len * 1.0 / len(dataset))
print("content_len: ", content_len * 1.0 / len(dataset))

cnt = 0
with codecs.open("xiaowei.online.62290", "w", encoding="utf8") as fw:
    for item in dataset:
        assert len(item["extract_sentence"]) > 0

        extract_sentence = "[SEP]".join(item["extract_sentence"])
        content = item["content"]
        corrected_sentence = answer_rewrite(extract_sentence, content)

        if extract_sentence != corrected_sentence:
            cnt += 1
            print(item["id"], extract_sentence, "#####", corrected_sentence)
        # "category": item["category"], "rewrite_sentence": item["rewrite_sentence"],
        json_str = {"id": item["id"], "title": item["title"],
                    "extract_sentence": corrected_sentence.split("[SEP]"),
                    "extract_label": item["extract_label"], "content": item["content"]}
        fw.write(json.dumps(json_str, ensure_ascii=False) + "\n")
print(cnt)  # 1556 6511
