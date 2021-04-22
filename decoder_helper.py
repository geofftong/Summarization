#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: geofftong
# Mail: geofftong@tencent.com
# Created Time: 2019/8/20 7:17 PM
import json


def _get_ngrams(n, text):
    """Calcualtes n-grams.

    Args:
      n: which n-grams to calculate
      text: An array of tokens

    Returns:
      A set of n-grams
    """
    ngram_set = set()
    text_length = len(text)
    max_index_ngram_start = text_length - n
    for i in range(max_index_ngram_start + 1):
        ngram_set.add(''.join(text[i:i + n]))  # tuple(text[i:i + n])
    # print(ngram_set)
    return ngram_set


def ngram_overlap(context, history_summary, n):
    """
    calculate ngram overlap between context and history summary
    :param context:
    :param history_summary:
    :return:
    """
    repeat_list = [0 for _ in context]
    repeat_pos_list = list()
    history_ngram_set = _get_ngrams(n, history_summary)
    # print(len(repeat_list), len(history_ngram_set))
    for pos, word in enumerate(context):
        cur_ngram = context[pos: pos + n]
        if cur_ngram in history_ngram_set:
            for idx in range(n):
                repeat_list[pos + idx] = 1
                if (pos + idx) not in repeat_pos_list:
                    repeat_pos_list.append(pos + idx)
    # print(repeat_list, repeat_pos_list)
    assert sum(repeat_list) == len(repeat_pos_list)
    return repeat_pos_list


def extract_label(result, context, previous_label, turn_idx):
    """
    find extracted sentence label given answer start position
    result:
    context:
    :return:
    """
    repeat_flag = False
    new_question = ""
    is_impossible = False
    text = result['text']
    start_idx = result['start_index']
    # if 'extract_label' in result:
    #     extract_label = result['extract_label']
    # else:
    #     extract_label = ["0" for _ in context]

    # turn_idx = max([int(label_str) for label_str in previous_label]) + 1  # 0902: bug fiexed
    if text == "":
        is_impossible = True
        # return new_question, extract_label, is_impossible, repeat_flag
        return new_question, previous_label, is_impossible, repeat_flag  # bug fixed: geoff 0820

    extract_state = ["0" for _ in context]
    idx, sent_idx = 0, 0
    # print(start_idx, len(extract_state), len(context))
    cnt = 0
    match_flag = False
    for idx, sentence in enumerate(context):
        if match_flag:
            break
        for _ in sentence:
            if int(start_idx) == cnt:
                extract_state[idx] = str(turn_idx)
                sent_idx = idx
                match_flag = True
                break
            cnt += 1

    # print(start_idx, sent_idx, text, context[sent_idx])
    assert text[0] in context[sent_idx]

    assert len(extract_state) == len(previous_label)
    for pos in range(len(previous_label)):
        if int(previous_label[pos]) > 0:
            if int(extract_state[pos]) > 0:
                repeat_flag = True
            else:
                extract_state[pos] = previous_label[pos]

    return context[sent_idx], extract_state, is_impossible, repeat_flag


def expand_nbest_file(predict_file, nbest_file, output_file, turn_idx, beam_size=5):
    """
    generate a new squad json file for next-turn prediction
    :param predict_file: qa4child.squad.dev
    :param nbest_file: nbest_predictions_1.json
    :param output_file: qa4child.squad.dev_2
    :param beam_size: top k
    :return:
    """
    qid_to_title = dict()
    qid_to_context = dict()
    qid_to_label = dict()  # qid to current label
    raw_data = json.load(open(predict_file))
    for item in raw_data['data']:
        qid = item['paragraphs'][0]['qas'][0]['id']
        qid_to_context[qid] = item['paragraphs'][0]['segmented_context']
        qid_to_title[qid] = item['paragraphs'][0]['qas'][0]['question']
        if 'extract_state' not in item['paragraphs'][0]['qas'][0]:
            qid_to_label[qid] = ["0" for _ in qid_to_context[qid]]
        else:
            qid_to_label[qid] = item['paragraphs'][0]['qas'][0]['extract_state']

    extract_state_dict = dict()
    expand_squad_data = list()

    # convert nbest_result in all beams to get top beam_size's prediction
    # nbest_json -> nbest_norm_json
    # input case 1: {'45': [top20]} -> {'45': [top3]}
    # input case 2: {'45-1': [top20]}, {'45-2': [top20]}, {'45-3': [top20]} -> {'45': [top3]}
    nbest_json = json.load(open(nbest_file))  # {qid: [top n result]}
    nbest_norm_json = dict()
    for qid, nbest_list in nbest_json.items():  # qid: 45-1
        root_id = qid.split("-")[0]
        for item in nbest_list:
            item['qid'] = qid  # add hash to news qid

        if root_id not in nbest_norm_json:  # 45
            nbest_norm_json[root_id] = nbest_list
        else:
            nbest_norm_json[root_id] += nbest_list

    print("##########sort in beam: ", len(nbest_json), len(nbest_norm_json))
    for qid, nbest_list in nbest_norm_json.items():  # qid: 45
        nbest_list = sorted(  # todo: score approach to be optimized
            nbest_list,
            key=lambda x: (x['start_logit'] + x['end_logit']),
            reverse=True)

        unique_sentences = set()  # filter repeat cases in beam size
        nbest_label = list()
        rank, loop_num = 0, 0
        # select n_best result
        last_extract_state = list()
        while rank < beam_size and loop_num < len(nbest_list):
            result_dict = nbest_list[loop_num]
            loop_num += 1
            raw_id = result_dict['qid']
            # result_dict: {"start_index":, "start_logit":, "end_index":, "end_logit":, "probility":, "text":, "qid":}
            # extract_state: used to evaluate
            # repeat_sent_flag: judge if current selected sentence is repeat
            # select_sentence: question
            # 0922: return current summary's length
            select_sentence, extract_state, is_impossible, repeat_sent_flag = extract_label(result_dict,
                                                                                            qid_to_context[raw_id],
                                                                                            qid_to_label[raw_id],
                                                                                          turn_idx)
            last_extract_state = extract_state
            # 0922: return current summary's length
            # cur_sum_len = 0
            # for sid, label in enumerate(extract_state):
            #     if int(label) > 0:
            #         cur_sum_len += len(qid_to_label[raw_id][sid])

            # 1021: bug fixed
            if not (len(nbest_label) == 0 and loop_num == len(nbest_list)):
                if repeat_sent_flag:
                    continue
                if select_sentence in unique_sentences:
                    continue
                # 0922: if current summary's length < k or turn_idx < 3, skipped no answer
                if is_impossible and turn_idx <= 3:
                    continue
            unique_sentences.add(select_sentence)

            if not is_impossible:
                select_sentence = qid_to_title[raw_id] + "[SEP]" + select_sentence
            else:
                select_sentence = qid_to_title[raw_id]
            expand_squad_data.append(
                [str(qid) + "-" + str(rank), extract_state, select_sentence, qid_to_context[raw_id]])

            nbest_label.append(extract_state)
            rank += 1

        # extract_state_dict[qid] = nbest_label[0]
        for idx, sorted_label in enumerate(nbest_label):  # geoff
            if "1" in sorted_label:
                extract_state_dict[qid] = sorted_label  # top 1
                break
            # if no answer in all beams, select first sentence
            if idx == len(nbest_label) - 1:
                extract_state_dict[qid] = nbest_label[0]
                extract_state_dict[qid][0] = "1"

    print("#######predict_label length: ", len(extract_state_dict))

    squad_json = convert_data_to_squad(expand_squad_data)
    json.dump(squad_json, open(output_file, "w"), ensure_ascii=False, sort_keys=True)
    return extract_state_dict


def convert_data_to_squad(raw_data):
    """
    raw_data:[[question, context], ...]
    :return:
    """
    mr_data = dict()
    mr_data["version"] = "v2.0"
    mr_data["data"] = list()

    for idx, item in enumerate(raw_data):
        qid, label, question, context = item
        samples = dict()
        samples["title"] = ""
        paragraph = dict()
        paragraph["qas"] = list()
        qas = dict()
        paragraph["context"] = "".join(context)
        paragraph["segmented_context"] = context  # geoff: diff from squad
        # paragraph["extract_state"] = label  # geoff: diff from squad
        qas["question"] = question
        qas["id"] = qid  # string
        qas["extract_state"] = label

        # 1021: ngram feature
        assert len(context) == len(label)
        history_summary = list()
        for pos, lab in enumerate(label):
            if int(lab) > 0:
                history_summary.append(context[pos])
        qas["repeat_3_gram"] = ngram_overlap("".join(context), "".join(history_summary), 3)
        qas["repeat_4_gram"] = ngram_overlap("".join(context), "".join(history_summary), 4)
        qas["repeat_5_gram"] = ngram_overlap("".join(context), "".join(history_summary), 5)
        # qas["repeat_6_gram"] = ngram_overlap("".join(context), "".join(history_summary), 6)
        # qas["repeat_7_gram"] = ngram_overlap("".join(context), "".join(history_summary), 7)

        paragraph["qas"] = [qas]
        samples["paragraphs"] = [paragraph]
        mr_data["data"].append(samples)
    return mr_data


def extract_span_label(result, context, previous_label):
    """
    find extracted sentence label given answer start position
    result: {"start_index":, "start_logit":, "end_index":, "end_logit":, "probility":, "text":, "qid":}
    context:
    :return:
    """
    repeat_flag, is_impossible = False, False
    new_question = ""
    text = result['text']
    start_idx, end_idx = result['start_index'], result['end_index']
    start_sent_idx, end_sent_idx = 0, 0

    if text == "":
        is_impossible = True
        # return new_question, extract_label, is_impossible, repeat_flag
        return new_question, previous_label, is_impossible, repeat_flag  # bug fixed: geoff 0820

    extract_state = ["0" for _ in context]
    # print(start_idx, len(extract_state), len(context))
    idx, cnt = 0, 0
    start_flag, end_flag = False, False
    for idx, sentence in enumerate(context):
        if end_flag:
            break
        for _ in sentence:
            if cnt == int(start_idx):
                extract_state[idx] = "1"
                start_sent_idx = idx
                start_flag = True
            if start_flag and cnt < int(end_idx):
                extract_state[idx] = "1"
            if cnt == int(end_idx):
                extract_state[idx] = "1"
                end_sent_idx = idx
                end_flag = True
                break
            cnt += 1

    # print(start_idx, sent_idx, text, context[sent_idx])
    assert text[0] in context[start_sent_idx]

    assert len(extract_state) == len(previous_label)
    for pos in range(len(previous_label)):
        if previous_label[pos] == '1':
            if extract_state[pos] == '1':
                repeat_flag = True
            else:
                extract_state[pos] = '1'

    return context[start_sent_idx: end_sent_idx + 1], extract_state, is_impossible, repeat_flag


def expand_nbest_span_file(predict_file, nbest_file, output_file, beam_size=5, max_history=1):
    """
    generate a new squad json file for next-turn prediction
    :param predict_file: qa4child.squad.dev
    :param nbest_file: nbest_predictions_1.json
    :param output_file: qa4child.squad.dev_2
    :param beam_size: top k
    :return:
    """
    qid_to_title = dict()
    qid_to_context = dict()
    qid_to_label = dict()  # qid to current label
    raw_data = json.load(open(predict_file))
    for item in raw_data['data']:
        qid = item['paragraphs'][0]['qas'][0]['id']
        qid_to_context[qid] = item['paragraphs'][0]['segmented_context']
        qid_to_title[qid] = item['paragraphs'][0]['qas'][0]['question']
        if 'extract_state' not in item['paragraphs'][0]['qas'][0]:
            qid_to_label[qid] = ["0" for _ in qid_to_context[qid]]
        else:
            qid_to_label[qid] = item['paragraphs'][0]['qas'][0]['extract_state']

    extract_state_dict = dict()
    expand_squad_data = list()

    # convert nbest_result in all beams to get top beam_size's prediction
    # nbest_json -> nbest_norm_json
    # input case 1: {'45': [top20]} -> {'45': [top3]}
    # input case 2: {'45-1': [top20]}, {'45-2': [top20]}, {'45-3': [top20]} -> {'45': [top3]}
    nbest_json = json.load(open(nbest_file))  # {qid: [top n result]}
    nbest_norm_json = dict()
    for qid, nbest_list in nbest_json.items():  # qid: 45-1
        root_id = qid.split("-")[0]
        for item in nbest_list:
            item['qid'] = qid  # add hash to news qid

        if root_id not in nbest_norm_json:  # 45
            nbest_norm_json[root_id] = nbest_list
        else:
            nbest_norm_json[root_id] += nbest_list

    print("##########sort in beam: ", len(nbest_json), len(nbest_norm_json))
    for qid, nbest_list in nbest_norm_json.items():  # qid: 45
        nbest_list = sorted(  # todo: score approach to be optimized
            nbest_list,
            key=lambda x: (x['start_logit'] + x['end_logit']),
            reverse=True)

        nbest_label = list()
        rank, loop_num = 0, 0
        while rank < beam_size and loop_num < len(nbest_list):
            result_dict = nbest_list[loop_num]
            loop_num += 1
            raw_id = result_dict['qid']
            # result_dict: {"start_index":, "start_logit":, "end_index":, "end_logit":, "probility":, "text":, "qid":}
            # extract_state: used to evaluate
            # repeat_sent_flag: true when sentence in current selected list is selected in history
            # select_sentence: question
            select_sentences_list, extract_state, is_impossible, repeat_sent_flag = extract_span_label(result_dict,
                                                                                                       qid_to_context[
                                                                                                           raw_id],
                                                                                                       qid_to_label[
                                                                                                           raw_id])
            if repeat_sent_flag:
                continue
            if extract_state in nbest_label:  # filter repeat cases in the same beam
                continue

            if not is_impossible:
                new_question = qid_to_title[raw_id].split("[SEP]")[0] + "[SEP]" + select_sentences_list[-1]
            else:
                new_question = qid_to_title[raw_id]
            expand_squad_data.append(
                [str(qid) + "-" + str(rank), extract_state, new_question, qid_to_context[raw_id]])

            nbest_label.append(extract_state)
            rank += 1
        # extract_state_dict[qid] = nbest_label[0]
        for idx, sorted_label in enumerate(nbest_label):  # geoff
            if "1" in sorted_label:
                extract_state_dict[qid] = sorted_label  # top 1
                break
            # if no answer in all beams, select first sentence
            if idx == len(nbest_label) - 1:
                extract_state_dict[qid] = nbest_label[0]
                extract_state_dict[qid][0] = "1"
    print("#######predict_label length: ", len(extract_state_dict))

    squad_json = convert_data_to_squad(expand_squad_data)
    json.dump(squad_json, open(output_file, "w"), ensure_ascii=False, sort_keys=True)
    return extract_state_dict
