#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: geofftong
# Mail: geofftong@tencent.com
# Created Time: 2019/8/20 7:17 PM
import json


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
    all_text = ''.join(context)
    start_idx = all_text.find(text)
    # if 'extract_label' in result:
    #     extract_label = result['extract_label']
    # else:
    #     extract_label = ["0" for _ in context]

    # turn_idx = max([int(label_str) for label_str in previous_label]) + 1  # 0902: bug fiexed
    if text == "" or start_idx == -1:
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


def expand_nbest_file(predict_file, nbest_file, null_odd_file,output_file, turn_idx, beam_size=5):
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
    qid2probs = {}
    raw_data = json.load(open(predict_file,'r',encoding='utf'))
    for item in raw_data['data']:
        qid = item['paragraphs'][0]['qas'][0]['id']
        qid_to_context[qid] = item['paragraphs'][0]['segmented_context']
        qid_to_title[qid] = item['paragraphs'][0]['qas'][0]['question']
        if 'extract_state' not in item['paragraphs'][0]['qas'][0]:
            qid_to_label[qid] = ["0" for _ in qid_to_context[qid]]
        else:
            qid_to_label[qid] = item['paragraphs'][0]['qas'][0]['extract_state']
        qid2probs[qid] = item['paragraphs'][0]['qas'][0].get('probability',1)

    extract_state_dict = dict()
    expand_squad_data = list()

    # convert nbest_result in all beams to get top beam_size's prediction
    # nbest_json -> nbest_norm_json
    # input case 1: {'45': [top20]} -> {'45': [top3]}
    # input case 2: {'45-1': [top20]}, {'45-2': [top20]}, {'45-3': [top20]} -> {'45': [top3]}
    nbest_json = json.load(open(nbest_file,'r',encoding='utf'))  # {qid: [top n result]}
    null_odd = json.load(open(null_odd_file,'r',encoding='utf'))
    nbest_norm_json = dict()
    for qid, nbest_list in nbest_json.items():  # qid: 45-1
        import math
        null_prob = 1 - 1/(math.exp(null_odd[qid])+1)
        null = {'text': '', 'probability': null_prob, 'start_log_prob': 0, 'end_log_prob': 0, 'qid': qid}
        nbest_list.append(null)
        root_id = qid.split("-")[0]
        for item in nbest_list:
            item['qid'] = qid  # add hash to news qid
            if item['text'] != '':
                item['probability'] *= 1 - null_prob
            item['probability'] *= qid2probs[qid]

        if root_id not in nbest_norm_json:  # 45
            nbest_norm_json[root_id] = nbest_list
        else:
            nbest_norm_json[root_id] += nbest_list

    print("##########sort in beam: ", len(nbest_json), len(nbest_norm_json))
    result_qid2probs = {}
    for qid, nbest_list in nbest_norm_json.items(): # qid: 45
        nbest_list = sorted(  # todo: score approach to be optimized
            nbest_list,
            key=lambda x: (x['probability']),
            reverse=True)

        unique_sentences = set()  # filter repeat cases in beam size
        nbest_label = list()
        rank, loop_num = 0, 0
        # select n_best result
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
            # 0922: return current summary's length
            # cur_sum_len = 0
            # for sid, label in enumerate(extract_state):
            #     if int(label) > 0:
            #         cur_sum_len += len(qid_to_label[raw_id][sid])
            if (loop_num == len(nbest_list) and rank == 0):
                if repeat_sent_flag :
                    is_impossible = True
                if select_sentence in unique_sentences :
                    is_impossible = True
            else:
                if repeat_sent_flag :
                    continue
                if select_sentence in unique_sentences :
                    continue
                # 0922: if current summary's length < k or turn_idx < 3, skipped no answer
                if is_impossible and turn_idx <= 3 :
                    continue
            unique_sentences.add(select_sentence)

            if not is_impossible:
                select_sentence = qid_to_title[raw_id] + "[SEP]" + select_sentence
            else:
                select_sentence = qid_to_title[raw_id]
            expand_squad_data.append(
                [str(qid) + "-" + str(rank), extract_state, select_sentence, qid_to_context[raw_id]])

            nbest_label.append(extract_state)
            result_qid2probs[str(qid) + "-" + str(rank)] = result_dict['probability']
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
    for item in squad_json['data']:
        qid = item['paragraphs'][0]['qas'][0]['id']
        item['paragraphs'][0]['qas'][0]['probability'] = result_qid2probs[qid]
    json.dump(squad_json, open(output_file, "w",encoding='utf'), ensure_ascii=False, sort_keys=True)
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
    raw_data = json.load(open(predict_file,'r',encoding='utf'))
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
    json.dump(squad_json, open(output_file, "w",encoding='utf'), ensure_ascii=False, sort_keys=True)
    return extract_state_dict
