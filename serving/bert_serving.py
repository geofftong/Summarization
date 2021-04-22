#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: geofftong
# Mail: geofftong@tencent.com
# Created Time: 2019/11/1 2:36 PM
import collections
import json
import os

import sys
import tensorflow as tf
import time
from serving_utils import convert_single_example, convert_example_to_features, post_process, update_result, \
    do_news_classify

import tokenization

reload(sys)
sys.setdefaultencoding('utf-8')

RawResult = collections.namedtuple("RawResult", ["unique_id", "start_logits", "end_logits"])


class NewsExtractor:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        self.tokenizer = tokenization.FullTokenizer(vocab_file=self.config["vocab_path"],
                                                    do_lower_case=self.config["do_lower_case"])
        self.predict_fn = tf.contrib.predictor.from_saved_model(self.config["model_path"])

    def __call__(self, news_title, news_content, news_category="", do_news_classify_flag=False):
        current_summary = []
        extract_state = []
        classify_result = [True, 1]  # "__label__适合",
        if do_news_classify_flag:
            classify_result = do_news_classify(self.config["classify_bin"], self.config["classify_model"], news_title,
                                               news_content, news_category)  # (label, score)
            if not classify_result[0]:  # == "__label__不适合"
                return "[SEP]".join(current_summary), classify_result[0], classify_result[1]

        for turn_idx in range(1, self.config["max_turn"] + 1):
            print(news_title, news_content)
            print(len(news_title), len(news_content))
            input_example = convert_single_example(news_title, news_content, extract_state)

            print("input example: ", input_example)

            input_features = convert_example_to_features(input_example, self.tokenizer, self.config["max_seq_length"],
                                                         self.config["doc_stride"], self.config["max_query_length"])
            if len(input_features) == 0:
                return "".join(current_summary), classify_result[0], classify_result[1]

            all_results = []
            for input_feature in input_features:
                output_prediction = self.predict_fn({
                    "input_ids": [input_feature.input_ids],
                    "input_mask": [input_feature.input_mask],
                    "segment_ids": [input_feature.segment_ids],
                    "unique_ids": [input_feature.unique_id],
                    'extract_state': [input_feature.extract_state],
                })

                all_results.append(
                    RawResult(
                        unique_id=int(output_prediction["unique_ids"]),
                        start_logits=[float(x) for x in output_prediction["start_logits"].flat],
                        end_logits=[float(x) for x in output_prediction["end_logits"].flat]))
            top_k_selections = post_process(input_example, input_features, all_results)

            news_title, news_content, extract_state, current_summary, stop_flag = update_result(input_example,
                                                                                                top_k_selections,
                                                                                                news_title,
                                                                                                news_content,
                                                                                                current_summary,
                                                                                                turn_idx)
            if stop_flag:
                break
        return "".join(current_summary), classify_result[0], classify_result[1]  # [SEP]


if __name__ == '__main__':
    start = time.time()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    ext = NewsExtractor(config_path="serving_config.json")

    news_category = "娱乐"
    news_title = "大标题: 周杰伦透露孩子性格像自己"
    # news_content = "自侃：在家地位不高周杰伦晒与儿子\\r\\n周杰伦与妻子昆凌\\r\\n新浪娱乐讯据台湾媒体报道，周杰伦今受邀出席公益篮球活动，许久没公开亮相的他坦言：“大家看我的社交网站就知道，现在玩乐和陪家人是最重要的。”他预计10月将启动全新巡演，但聊到新专辑，他笑说：“目前已经做好5首，但包括上次做好的2首。”进度严重落后，巡演前能否来得及推出仍是未知数。\\r\\n周杰伦曾被“老萧”萧敬腾与五月天阿信调侃：“巴黎铁塔好像是他家一样”，周杰伦也坦言有考虑在法国置产，地点属意：“有巴黎铁塔蛮重要的”，他感叹身处欧洲比较可以自由自在走在街上，就算被粉丝认出，打个招呼就滑着滑板溜走，不用跑得太狼狈。\\r\\n今天他与小朋友共享篮球时光，但聊到自己的一双儿女，他说：“小朋友个性像自己，比较顽固一点，小朋友都是这样、比较讲不听，严格来说我扮黑脸也是没什么用，在家里地位不是很高。”他形容女儿就像自己“另个女朋友”，有时候想和她和好还被拒绝，一直被闹别扭，周杰伦无奈说：“我还不知道怎么对待一个女儿。”倒是儿子比较怕他，只要一出声就会低头认错，“像他会画在桌子上，家里很多画，就会教他看画手要放在后面，不要摸”严父模样只有儿子会买单。\\r\\n阿信曾夸周杰伦是“华人音乐圈精神领袖”，周杰伦赞阿信是“心目中真正的音乐人”，曾邀阿信到家中作客，2人畅谈音乐，阿信还精辟分析他专辑，“发现他是实际有在听我歌曲的人”，但问到每次打球都只邀萧敬腾，周杰伦笑说有透过社交网站约阿信打球，每次却只被回表情符号，忍不住说：“他说他以前打曲棍球，我是不太相信的。”\\r\\n周杰伦将于10月启动全新巡演，谈到近期的困扰，就是还没想到一个很威的名字，“想不到比‘地表最强’更好的，但这次跟（出道）20年有关，会比较欢乐，不是走自己多强势的感觉”。\\r\\n(责编：漠er凡)\\r\\n"
    news_content = " "
    result, label, score = ext(news_title=news_title.decode("utf-8"), news_category=news_category.decode("utf8"),
                               news_content=news_content.decode("utf-8"),
                               do_news_classify_flag=True)
    print("final output: %s" % "".join(result))
    print(label, score)

    end = time.time()
    print("time: %s" % (end - start))
