#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: geofftong
# Mail: geofftong@tencent.com
# Created Time: 2019/11/5 2:36 PM
import collections
import copy

import math
import six
import os
import tokenization


class SquadExample(object):
    """A single training/test example for simple sequence classification.
       For examples without an answer, the start and end position are -1.
    """

    def __init__(self,
                 qas_id,
                 question_text,
                 doc_tokens,
                 extract_state,
                 seg_doc,
                 orig_answer_text=None,
                 start_position=None,
                 end_position=None,
                 is_impossible=False):
        self.qas_id = qas_id
        self.question_text = question_text
        self.doc_tokens = doc_tokens
        self.extract_state = extract_state
        self.seg_doc = seg_doc
        self.orig_answer_text = orig_answer_text
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible


class InputFeatures(object):
    """A single set of features of data_dir."""

    def __init__(self,
                 unique_id,
                 example_index,
                 doc_span_index,
                 tokens,
                 token_to_orig_map,
                 token_is_max_context,
                 input_ids,
                 input_mask,
                 segment_ids,
                 extract_state,
                 start_position=None,
                 end_position=None,
                 is_impossible=None):
        self.unique_id = unique_id
        self.example_index = example_index
        self.doc_span_index = doc_span_index
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map
        self.token_is_max_context = token_is_max_context
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible
        self.extract_state = extract_state


def convert_single_example(news_title, news_content, extract_state):
    # norm chinese punctuation
    # news_title = _half_to_full(news_title)
    # news_content = _half_to_full(news_content)

    doc_tokens = []
    for c in news_content.replace("\\r\\n", ""):
        doc_tokens.append(c)

    segmented_doc = []
    for para in news_content.split("\\r\\n"):
        if len(para.strip()) != 0:
            segmented_doc.extend(_ch_sent_split(para.strip()))

    # segmented_doc = _ch_sent_split(news_content)

    if len(extract_state) == 0:  # first turn
        extract_state = [0 for _ in doc_tokens]

    # for sent_idx, state in enumerate(extract_state):
    #     if state == 0:
    #         expand_extract_state.extend([0 for _ in range(len(segmented_doc[sent_idx]))])
    #     else:
    #         expand_extract_state.extend([1 for _ in range(len(segmented_doc[sent_idx]))])
    assert len(extract_state) == len(doc_tokens)
    print("doc_tokens: ", len(doc_tokens))

    example = SquadExample(
        qas_id=0,
        question_text=news_title,
        doc_tokens=doc_tokens,
        seg_doc=segmented_doc,
        extract_state=extract_state,  # len(news_state) == len(paragraph_text)
        orig_answer_text=None,
        start_position=None,
        end_position=None,
        is_impossible=False)
    return example


def _half_to_full(context):
    context = context.replace('“', '\"').replace('”', '\"')
    context = context.replace(';', '；').replace('!', '！'). \
        replace('?', '？').replace('(', '（'). \
        replace(')', '）')  # .replace(' ', '').replace('.', '。')
    return context


def _ch_sent_split(content, high_recall=True):
    """
    中文断句，remains punctuation
    :param content:
    :param high_recall: 允许最后一句不以标点符号结尾
    :return:
    """
    # content = content.replace(";", "；").replace("!", "！").replace("?", "？")
    all_sentence = []
    sent_cut_punc = ["。", "！", "？", "!", "?"]
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


def convert_example_to_features(input_example, tokenizer, max_seq_length, doc_stride, max_query_length):
    unique_id = 1000000000
    features = []
    query_tokens = []
    all_extract_state = input_example.extract_state
    for segment in input_example.question_text.split("[SEP]"):
        query_tokens.extend(tokenizer.tokenize(segment))
        query_tokens.append("[SEP]")  # optional
    query_tokens = query_tokens[:-1]
    if len(query_tokens) > max_query_length:
        query_tokens = query_tokens[0:max_query_length]

    tok_to_orig_index = []
    orig_to_tok_index = []
    all_doc_tokens = []
    for (i, token) in enumerate(input_example.doc_tokens):
        orig_to_tok_index.append(len(all_doc_tokens))
        sub_tokens = tokenizer.tokenize(token)
        for sub_token in sub_tokens:
            tok_to_orig_index.append(i)
            all_doc_tokens.append(sub_token)

    # The -3 accounts for [CLS], [SEP] and [SEP]
    max_tokens_for_doc = max_seq_length - len(query_tokens) - 3  # char

    # create all spans for a context
    _DocSpan = collections.namedtuple("DocSpan", ["start", "length"])
    doc_spans = []
    start_offset = 0

    while start_offset < len(all_doc_tokens):
        length = len(all_doc_tokens) - start_offset
        if length > max_tokens_for_doc:
            length = max_tokens_for_doc
        doc_spans.append(_DocSpan(start=start_offset, length=length))  # char
        if start_offset + length == len(all_doc_tokens):
            break
        start_offset += min(length, doc_stride)

    for (doc_span_index, doc_span) in enumerate(doc_spans):  # time costing
        tokens = []
        token_to_orig_map = {}  # doc tokens position in all input tokens[18:0 19:1 ...] [18:257 19:258 ...]
        token_is_max_context = {}
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in query_tokens:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)
        # convert span extract state
        span_extract_state = [0] * len(segment_ids)

        for i in range(doc_span.length):
            span_extract_state.append(all_extract_state[doc_span.start + i])
            split_token_index = doc_span.start + i
            token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]
            is_max_context = _check_is_max_context(doc_spans, doc_span_index,
                                                   split_token_index)
            token_is_max_context[len(tokens)] = is_max_context
            tokens.append(all_doc_tokens[split_token_index])
            segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)
        span_extract_state.append(0)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            span_extract_state.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(span_extract_state) == max_seq_length

        feature = InputFeatures(
            unique_id=unique_id,
            example_index=0,
            doc_span_index=doc_span_index,
            tokens=tokens,
            token_to_orig_map=token_to_orig_map,
            token_is_max_context=token_is_max_context,
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids,
            start_position=None,
            end_position=None,
            is_impossible=False,
            extract_state=span_extract_state)
        features.append(feature)
        unique_id += 1
    return features


def _check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""

    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index
    return cur_span_index == best_span_index


def post_process(input_example, all_features, all_results, n_best_size=20, max_answer_length=120, do_lower_case=True):
    """Write final predictions to the json file and log-odds of null if needed."""

    example_index_to_features = collections.defaultdict(list)

    print("len of all features: ", len(all_features))
    for feature in all_features:
        example_index_to_features["0"].append(feature)  # example_index = 0

    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    _PrelimPrediction = collections.namedtuple("PrelimPrediction",
                                               ["feature_index", "start_index", "end_index", "start_logit",
                                                "end_logit"])

    print("len of example_index_to_features: ", len(example_index_to_features))
    features = example_index_to_features["0"]  # feature list in the same example
    print("features: ", features)

    prelim_predictions = []
    # keep track of the minimum score of null start+end of position 0
    score_null = 1000000  # large and positive
    min_null_feature_index = 0  # the paragraph slice with min mull score
    null_start_logit = 0  # the start logit at the slice with min null score
    null_end_logit = 0  # the end logit at the slice with min null score
    for (feature_index, feature) in enumerate(features):
        result = unique_id_to_result[feature.unique_id]
        start_indexes = _get_best_indexes(result.start_logits, n_best_size)
        end_indexes = _get_best_indexes(result.end_logits, n_best_size)
        # if we could have irrelevant answers, get the min score of irrelevant
        feature_null_score = result.start_logits[0] + result.end_logits[0]
        if feature_null_score < score_null:
            score_null = feature_null_score
            min_null_feature_index = feature_index
            null_start_logit = result.start_logits[0]
            null_end_logit = result.end_logits[0]

        for start_index in start_indexes:
            for end_index in end_indexes:
                if start_index >= len(feature.tokens):
                    continue
                if end_index >= len(feature.tokens):
                    continue
                if start_index not in feature.token_to_orig_map:
                    continue
                if end_index not in feature.token_to_orig_map:
                    continue
                if not feature.token_is_max_context.get(start_index, False):
                    continue
                if end_index < start_index:
                    continue
                length = end_index - start_index + 1
                if length > max_answer_length:
                    continue
                prelim_predictions.append(
                    _PrelimPrediction(
                        feature_index=feature_index,
                        start_index=start_index,
                        end_index=end_index,
                        start_logit=result.start_logits[start_index],
                        end_logit=result.end_logits[end_index]))

    prelim_predictions.append(
        _PrelimPrediction(
            feature_index=min_null_feature_index,
            start_index=0,
            end_index=0,
            start_logit=null_start_logit,
            end_logit=null_end_logit))

    prelim_predictions = sorted(
        prelim_predictions,
        key=lambda x: (x.start_logit + x.end_logit),
        reverse=True)

    _NbestPrediction = collections.namedtuple(
        "NbestPrediction",
        ["text", "start_index", "end_index", "start_logit", "end_logit"])  # geoff: add index to detect sentence

    seen_predictions = {}
    nbest = []
    for pred in prelim_predictions:
        if len(nbest) >= n_best_size:
            break
        feature = features[pred.feature_index]
        # if pred.start_index > 0:  # this is a non-null prediction
        if pred.start_index > 0 or (pred.start_index == 0 and pred.end_index > 0):  # geoff: bert bug fixed
            tok_tokens = feature.tokens[pred.start_index:(pred.end_index + 1)]
            orig_doc_start = feature.token_to_orig_map[pred.start_index]
            orig_doc_end = feature.token_to_orig_map[pred.end_index]
            orig_tokens = input_example.doc_tokens[orig_doc_start:(orig_doc_end + 1)]
            tok_text = " ".join(tok_tokens)

            # De-tokenize WordPieces that have been split off.
            tok_text = tok_text.replace(" ##", "")
            tok_text = tok_text.replace("##", "")

            # Clean whitespace
            tok_text = tok_text.strip()
            tok_text = " ".join(tok_text.split())
            orig_text = " ".join(orig_tokens)

            final_text = get_final_text(tok_text, orig_text, do_lower_case)
            if final_text in seen_predictions:
                continue

            seen_predictions[final_text] = True
        else:
            final_text = ""
            seen_predictions[final_text] = True
            orig_doc_start = 0
            orig_doc_end = 0

        nbest.append(
            _NbestPrediction(
                text=final_text,
                start_index=orig_doc_start,
                end_index=orig_doc_end,
                start_logit=pred.start_logit,
                end_logit=pred.end_logit))

    # if we didn't inlude the empty option in the n-best, inlcude it
    if "" not in seen_predictions:
        nbest.append(
            _NbestPrediction(
                text="", start_index=0, end_index=0, start_logit=null_start_logit,
                end_logit=null_end_logit))
    # In very rare edge cases we could have no valid predictions. So we
    # just create a nonce prediction in this case to avoid failure.
    if not nbest:
        nbest.append(
            _NbestPrediction(text="", start_index=0, end_index=0, start_logit=0.0, end_logit=0.0))  # "empty"

    total_scores = []
    best_non_null_entry = None
    for entry in nbest:
        total_scores.append(entry.start_logit + entry.end_logit)
        if not best_non_null_entry:
            if entry.text:
                best_non_null_entry = entry

    probs = _compute_softmax(total_scores)

    nbest_json = []
    for (i, entry) in enumerate(nbest):  # geoff: start_idx -> sentence_label
        output = collections.OrderedDict()
        output["text"] = entry.text
        output["probability"] = probs[i]
        output["start_index"] = entry.start_index
        output["end_index"] = entry.end_index
        output["start_logit"] = entry.start_logit
        output["end_logit"] = entry.end_logit
        nbest_json.append(output)

    return nbest_json  # top_k_selections


def get_final_text(pred_text, orig_text, do_lower_case):
    """Project the tokenized prediction back to the original text."""

    def _strip_spaces(text):
        ns_chars = []
        ns_to_s_map = collections.OrderedDict()
        for (i, c) in enumerate(text):
            if c == " ":
                continue
            ns_to_s_map[len(ns_chars)] = i
            ns_chars.append(c)
        ns_text = "".join(ns_chars)
        return (ns_text, ns_to_s_map)

    # We first tokenize `orig_text`, strip whitespace from the result
    # and `pred_text`, and check if they are the same length. If they are
    # NOT the same length, the heuristic has failed. If they are the same
    # length, we assume the characters are one-to-one aligned.
    tokenizer = tokenization.BasicTokenizer(do_lower_case=do_lower_case)

    tok_text = " ".join(tokenizer.tokenize(orig_text))

    start_position = tok_text.find(pred_text)
    if start_position == -1:
        print("Unable to find text: '%s' in '%s'" % (pred_text, orig_text))
        return orig_text
    end_position = start_position + len(pred_text) - 1

    (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
    (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

    if len(orig_ns_text) != len(tok_ns_text):
        print("Length not equal after stripping spaces: '%s' vs '%s'",
              orig_ns_text, tok_ns_text)
        return orig_text

    # We then project the characters in `pred_text` back to `orig_text` using
    # the character-to-character alignment.
    tok_s_to_ns_map = {}
    for (i, tok_index) in six.iteritems(tok_ns_to_s_map):
        tok_s_to_ns_map[tok_index] = i

    orig_start_position = None
    if start_position in tok_s_to_ns_map:
        ns_start_position = tok_s_to_ns_map[start_position]
        if ns_start_position in orig_ns_to_s_map:
            orig_start_position = orig_ns_to_s_map[ns_start_position]

    if orig_start_position is None:
        print("Couldn't map start position")
        return orig_text

    orig_end_position = None
    if end_position in tok_s_to_ns_map:
        ns_end_position = tok_s_to_ns_map[end_position]
        if ns_end_position in orig_ns_to_s_map:
            orig_end_position = orig_ns_to_s_map[ns_end_position]

    if orig_end_position is None:
        print("Couldn't map end position")
        return orig_text

    output_text = orig_text[orig_start_position:(orig_end_position + 1)]
    return output_text


def _get_best_indexes(logits, n_best_size):
    """Get the n-best logits from a list."""
    index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

    best_indexes = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])
    return best_indexes


def _compute_softmax(scores):
    """Compute softmax probability over news logits."""
    if not scores:
        return []

    max_score = None
    for score in scores:
        if max_score is None or score > max_score:
            max_score = score

    exp_scores = []
    total_sum = 0.0
    for score in scores:
        x = math.exp(score - max_score)
        exp_scores.append(x)
        total_sum += x

    probs = []
    for score in exp_scores:
        probs.append(score / total_sum)
    return probs


def update_result(input_example, nbest_json, news_title, news_content, current_summary, turn_idx, beam_size=1):
    """ return top 1 result (instead of top k) """
    null_answer_flag = False
    nbest_list = sorted(
        nbest_json,
        key=lambda x: (x['start_logit'] + x['end_logit']),
        reverse=True)
    expand_beam_data = list()
    unique_sentences = set()
    rank, loop_num = 0, 0
    while rank < beam_size and loop_num < len(nbest_list):
        result_dict = nbest_list[loop_num]
        loop_num += 1

        select_sentence, extract_state, is_impossible, repeat_sent_flag = _extract_label(result_dict,
                                                                                         input_example.seg_doc,
                                                                                         input_example.extract_state,
                                                                                         turn_idx)
        if not (rank == 0 and loop_num == len(nbest_list)):
            if repeat_sent_flag:  # repeat result in history summary
                continue
            if select_sentence in unique_sentences:  # repeat result in beams
                continue
            # if (current summary's length < L or) turn_idx < 3, skipped no answer
            if is_impossible and turn_idx <= 3:
                continue
        unique_sentences.add(select_sentence)
        if not is_impossible:
            news_title = news_title + "[SEP]" + select_sentence  # rewrite text_a

        expand_beam_data.append(
            [str(rank), extract_state, news_title, news_content])
        if len(select_sentence) > 0:
            current_summary.append(select_sentence)
        rank += 1

    if len(expand_beam_data) == 0:
        null_answer_flag = True
        expand_beam_data.append((news_title, news_content, input_example.extract_state, current_summary))
    return expand_beam_data[0][2], expand_beam_data[0][3], expand_beam_data[0][1], current_summary, null_answer_flag


def _extract_label(result, context, extract_state, turn_idx):
    """ update extract label by start_idx of answers"""

    repeat_flag = False
    is_impossible = False
    text = result['text']
    start_idx = result['start_index']
    if text == "":
        return "", extract_state, True, False

    cnt, sent_start_pos, sent_end_pos, sent_idx = 0, 0, 0, 0
    match_flag = False
    for idx, sentence in enumerate(context):
        if match_flag:
            break
        sent_start_pos = cnt
        for _ in sentence:
            if int(start_idx) == cnt:
                sent_idx = idx
                match_flag = True
            cnt += 1
        sent_end_pos = cnt
    if extract_state[start_idx] == 1:
        repeat_flag = True
    else:
        for pos in range(sent_start_pos, sent_end_pos):
            extract_state[pos] = 1

    return context[sent_idx], extract_state, is_impossible, repeat_flag


def do_news_classify(fast_bin, fast_model, news_title, news_content, news_category=""):
    unused_category = ["旅游", "时尚", "菜谱", "评测"]
    # unused_sub_category = ["美食_菜谱", "汽车_汽车评测"]
    unused_keywords = ["财经24小时", "新闻汇总", "新闻汇编", "征集违法犯罪线索", "招生说明", "失物招领", "通知", "通告", "启事"]

    for word in unused_keywords:
        if word in news_title:  # news_title.endswith(word)
            return False, 1.0  # "__label__不适合"

    for cate in unused_category:
        if cate in news_category:
            return False, 1.0

    # call fasttext classifier binary
    fast_input = " ".join([_ for _ in news_category]) + " " + " ".join([_ for _ in news_title]) + " " + " ".join([_ for _ in news_content])
    call_cmd = "./{} {} '{}'".format(fast_bin, fast_model, fast_input.replace("'", ""))
    with os.popen(call_cmd, 'r') as f:
        result = f.read()
    label, score = result.strip().split("\t")
    if label == "__label__适合":
        return True, float(score)
    else:
        return False, float(score)
