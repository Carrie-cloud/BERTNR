from __future__ import absolute_import, division, print_function

import argparse
import csv
import logging
import os
import random
import sys

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from torch.nn import CrossEntropyLoss, MSELoss
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score
from sklearn import metrics
from sklearn.metrics import precision_recall_curve, auc
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.modeling import BertForSequenceClassification, BertConfig
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule

os.environ['CUDA_VISIBCLE_DEVICES'] = '0'

logger = logging.getLogger(__name__)
print(torch.cuda.is_available())

class InputExample(object):

    def __init__(self, guid, text_a, text_b=None, text_c=None, label=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.text_c = text_c
        self.label = label

    def __eq__(self, other):
        return (self.text_a, self.text_b, self.text_c, self.label) == (
        other.text_a, other.text_b, other.text_c, other.label)

    def __hash__(self):
        return hash((self.text_a, self.text_b, self.text_c, self.label))


class InputFeatures(object):

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self, data_dir):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(str(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines


class KGProcessor(DataProcessor):
    """Processor for knowledge graph data set."""

    def __init__(self):
        self.labels = set()

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.txt")), "train", data_dir)

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.txt")), "dev", data_dir)

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.txt")), "test", data_dir)

    def get_relations(self, data_dir):
        """Gets all labels (relations) in the knowledge graph."""
        # return list(self.labels)
        with open(os.path.join(data_dir, "relations.txt"), 'r') as f:
            lines = f.readlines()
            relations = []
            for line in lines:
                relations.append(line.strip())
        return relations

    def get_labels(self, data_dir):
        """Gets all labels (0, 1) for triples in the knowledge graph."""
        return ["0", "1"]

    def get_entities(self, data_dir):
        """Gets all entities in the knowledge graph."""
        # return list(self.labels)
        with open(os.path.join(data_dir, "entities.txt"), 'r') as f:
            lines = f.readlines()
            entities = []
            for line in lines:
                entities.append(line.strip())
        return entities

    def get_train_triples(self, data_dir):
        """Gets training triples."""
        return self._read_tsv(os.path.join(data_dir, "train.txt"))

    def get_dev_triples(self, data_dir):
        """Gets validation triples."""
        return self._read_tsv(os.path.join(data_dir, "dev.txt"))

    def get_test_triples(self, data_dir):
        """Gets test triples."""
        return self._read_tsv(os.path.join(data_dir, "test.txt"))

    def get_train_entities(self, data_dir):

        with open(os.path.join(data_dir, "train.txt"), mode='r', encoding='utf-8') as file:
            tsv_reader = list(csv.reader(file, delimiter='\t'))

        # 初始化已访问元素集合
        visited_elements = set()
        # 提取已访问元素集合
        for row in tsv_reader:
            visited_elements.add(row[0])
            visited_elements.add(row[2])
        return visited_elements

    def get_dev_entities(self, data_dir):

        with open(os.path.join(data_dir, "dev.txt"), mode='r', encoding='utf-8') as file:
            tsv_reader = list(csv.reader(file, delimiter='\t'))

        # 初始化已访问元素集合
        visited_elements = set()
        # 提取已访问元素集合
        for row in tsv_reader:
            visited_elements.add(row[0])
            visited_elements.add(row[2])
        return visited_elements

    def get_test_entities(self, data_dir):

        with open(os.path.join(data_dir, "test.txt"), mode='r', encoding='utf-8') as file:
            tsv_reader = list(csv.reader(file, delimiter='\t'))

        # 初始化已访问元素集合
        visited_elements = set()
        # 提取已访问元素集合
        for row in tsv_reader:
            visited_elements.add(row[0])
            visited_elements.add(row[2])
        return visited_elements

    def get_relation(self, data_dir):

        rel2text = {}
        with open(os.path.join(data_dir, "relation2text.txt"), 'r') as file:
            tsv_reader = list(csv.reader(file, delimiter='\t'))

        # 初始化已访问元素集合
        visited_elements = set()
        # 提取已访问元素集合
        for row in tsv_reader:
            visited_elements.add(row[1])
        return visited_elements

    def read_and_parse_trainfile(self, file_path):
        parsed_data = {}
        with open(os.path.join(file_path, "TNedge.tsv"), 'r', encoding='utf-8') as file:
            for line in file:
                parts = line.strip().split(',')
                key = parts[0]
                values = parts[1:]
                parsed_data[key] = []
                for i in range(len(values)):
                    parsed_data[key].append(values[i])
        # print(len(parsed_data))
        return parsed_data

    def read_and_parse_testfile(self, file_path):
        parsed_data = {}
        with open(os.path.join(file_path, "TEedge.tsv"), 'r', encoding='utf-8') as file:
            for line in file:
                parts = line.strip().split(',')
                key = parts[0]
                values = parts[1:]
                parsed_data[key] = []
                for i in range(len(values)):
                    parsed_data[key].append(values[i])
        # print(len(parsed_data))
        return parsed_data

    def read_and_parse_TNoutfile(self, file_path):
        parsed_data = {}
        with open(os.path.join(file_path, "TNout.tsv"), 'r', encoding='utf-8') as file:
            for line in file:
                parts = line.strip().split(',')
                key = parts[0]
                values = parts[1:]
                parsed_data[key] = []
                for i in range(len(values)):
                    parsed_data[key].append(values[i])
        # print(len(parsed_data))
        return parsed_data

    def read_and_parse_TNinfile(self, file_path):
        parsed_data = {}
        with open(os.path.join(file_path, "TNedge.tsv"), 'r', encoding='utf-8') as file:
            for line in file:
                parts = line.strip().split(',')
                key = parts[0]
                values = parts[1:]
                parsed_data[key] = []
                for i in range(len(values)):
                    parsed_data[key].append(values[i])
        # print(len(parsed_data))
        return parsed_data

    def read_and_parse_TEoutfile(self, file_path):
        parsed_data = {}
        with open(os.path.join(file_path, "TEout.tsv"), 'r', encoding='utf-8') as file:
            for line in file:
                parts = line.strip().split(',')
                key = parts[0]
                values = parts[1:]
                parsed_data[key] = []
                for i in range(len(values)):
                    parsed_data[key].append(values[i])
        # print(len(parsed_data))
        return parsed_data

    def read_and_parse_TEinfile(self, file_path):
        parsed_data = {}
        with open(os.path.join(file_path, "TEin.tsv"), 'r', encoding='utf-8') as file:
            for line in file:
                parts = line.strip().split(',')
                key = parts[0]
                values = parts[1:]
                parsed_data[key] = []
                for i in range(len(values)):
                    parsed_data[key].append(values[i])
        # print(len(parsed_data))
        return parsed_data

    def process_list_items(self, parsed_data, search_key, label, rel2text):
        realist = ""
        # Check if the search key is in the parsed data
        if search_key in parsed_data:
            # print(f"Items for*****'{search_key}'******:")
            for item in parsed_data[search_key]:
                if item != label:
                    realist += item
                    realist += " "

            # print(f"******{search_key} {len(realist)}******")
        return realist

    def _create_examples(self, lines, set_type, data_dir):

        """Creates examples for the training and dev sets."""
        # entity to text
        ent2text = {}
        with open(os.path.join(data_dir, "entity2text.txt"), 'r', encoding='utf-8') as f:
            ent_lines = f.readlines()
            for line in ent_lines:
                temp = line.strip().split('\t')
                if len(temp) == 2:
                    ent2text[temp[0]] = temp[1]

        # relation to text
        rel2text = {}
        with open(os.path.join(data_dir, "relation2text.txt"), 'r', encoding='utf-8') as f:
            rel_lines = f.readlines()
            for line in rel_lines:
                temp = line.strip().split('\t')
                rel2text[temp[0]] = temp[1]

        lines_str_set = set(['\t'.join(line) for line in lines])
        examples = []
        for (i, line) in enumerate(lines):
            head_ent_text = ent2text[line[0]]
            tail_ent_text = ent2text[line[2]]
            relation_text = rel2text[line[1]]

            if set_type == "train":
                # parsed_dataout = self.read_and_parse_TNoutfile(data_dir)
                # parsed_datain = self.read_and_parse_TNinfile(data_dir)
                parsed_data = self.read_and_parse_trainfile(data_dir)
                guid = "%s-%s" % (set_type, i)
                text_a = head_ent_text
                text_b = tail_ent_text
                text_c = relation_text
                entities = self.get_train_entities(data_dir)
                re_texta = ""
                search_key = line[0]  # 可以替换为您想要搜索的键
                # 使用之前解析的数据
                contextlist = self.process_list_items(parsed_data, search_key, line[1], rel2text)
                re_texta += (" " + contextlist)

                re_textb = ""
                search_key = line[2]  # 可以替换为您想要搜索的键
                # 使用之前解析的数据
                contextlist = self.process_list_items(parsed_data, search_key, line[1], rel2text)
                re_textb += (" " + contextlist)
                examples.append(
                    InputExample(guid=guid, text_a=text_c, text_b= re_texta + re_textb, text_c= text_a + text_b,
                                 label="1"))

                rnd = random.random()
                guid = "%s-%s" % (set_type + "_corrupt", i)
                if rnd <= 0.5:
                    # corrupting head
                    for j in range(1):
                        tmp_head = ''
                        while True:
                            tmp_ent_list = set(entities)
                            tmp_ent_list.remove(line[0])
                            tmp_ent_list = list(tmp_ent_list)
                            tmp_head = random.choice(tmp_ent_list)
                            tmp_triple_str = tmp_head + '\t' + line[1] + '\t' + line[2]
                            if tmp_triple_str not in lines_str_set:
                                break
                        tmp_head_text = ent2text[tmp_head]
                        text_a = tmp_head_text
                        re_texta = ""
                        search_key = tmp_head
                        contextlist = self.process_list_items(parsed_data, search_key, line[1], rel2text)
                        re_texta += (" " + contextlist)
                        examples.append(
                            InputExample(guid=guid, text_a=text_c, text_b= re_texta + re_textb, text_c= text_a + text_b,
                                         label="0"))
                else:
                    # corrupting tail
                    tmp_tail = ''
                    for j in range(1):
                        while True:
                            tmp_ent_list = set(entities)
                            tmp_ent_list.remove(line[2])
                            tmp_ent_list = list(tmp_ent_list)
                            tmp_tail = random.choice(tmp_ent_list)
                            tmp_triple_str = line[0] + '\t' + line[1] + '\t' + tmp_tail
                            if tmp_triple_str not in lines_str_set:
                                break
                        tmp_tail_text = ent2text[tmp_tail]
                        text_b = tmp_tail_text
                        re_textb = ""
                        search_key = tmp_tail  # 可以替换为您想要搜索的键
                        # 使用之前解析的数据
                        contextlist = self.process_list_items(parsed_data, search_key, line[1], rel2text)
                        re_textb += (" " + contextlist)
                        examples.append(
                            InputExample(guid=guid, text_a=text_c, text_b= re_texta + re_textb, text_c= text_a + text_b,
                                         label="0"))
        return examples

    def create_examples_negative(self, test_triple, set_type, data_dir, lines_str_set):

        """Creates examples for the training and dev sets."""
        # entity to text
        ent2text = {}
        with open(os.path.join(data_dir, "entity2text.txt"), 'r', encoding='utf-8') as f:
            ent_lines = f.readlines()
            for line in ent_lines:
                temp = line.strip().split('\t')
                if len(temp) == 2:
                    ent2text[temp[0]] = temp[1]

        # relation to text
        rel2text = {}
        with open(os.path.join(data_dir, "relation2text.txt"), 'r', encoding='utf-8') as f:
            rel_lines = f.readlines()
            for line in rel_lines:
                temp = line.strip().split('\t')
                rel2text[temp[0]] = temp[1]

        examples = []

        head_ent_text = ent2text[test_triple[0]]
        tail_ent_text = ent2text[test_triple[2]]
        relation_text = rel2text[test_triple[1]]

        if set_type == "dev" or set_type == "test":
            # parsed_dataout = self.read_and_parse_TEoutfile(data_dir)
            # parsed_datain = self.read_and_parse_TEinfile(data_dir)
            parsed_data = self.read_and_parse_testfile(data_dir)
            guid = "%s-%s" % (set_type, 0)
            text_a = head_ent_text
            text_b = tail_ent_text
            text_c = relation_text
            re_texta = ""
            search_key = test_triple[0]  # 可以替换为您想要搜索的键
            # 使用之前解析的数据
            contextlist = self.process_list_items(parsed_data, search_key, test_triple[1], rel2text)
            re_texta += (" " + contextlist)

            re_textb = ""
            search_key = test_triple[2]  # 可以替换为您想要搜索的键
            # 使用之前解析的数据
            contextlist = self.process_list_items(parsed_data, search_key, test_triple[1], rel2text)
            re_textb += (" " + contextlist)

            examples.append(
                InputExample(guid=guid, text_a=text_c, text_b= re_texta + re_textb, text_c= text_a + text_b,
                             label="1"))

            entities = self.get_test_entities(data_dir)
            k = 1
            # 生成50个负样本
            examples_negative = []
            while len(examples_negative) < 50:
                rnd = random.random()
                guid = "%s-%s" % (set_type + "_corrupt", k)
                k = k + 1
                if rnd <= 0.5:
                    # corrupting head
                    while True:
                        tmp_ent_list = set(entities)
                        tmp_ent_list.remove(test_triple[0])
                        tmp_ent_list = list(tmp_ent_list)
                        tmp_head = random.choice(tmp_ent_list)
                        tmp_triple_str = tmp_head + '\t' + test_triple[1] + '\t' + test_triple[2]
                        if tmp_triple_str not in lines_str_set:
                            break
                    tmp_head_text = ent2text[tmp_head]
                    text_a = tmp_head_text
                    re_texta = ""
                    search_key = tmp_head
                    contextlist = self.process_list_items(parsed_data, search_key, test_triple[1], rel2text)
                    re_texta += (" " + contextlist)
                    examples_negative.append(
                        InputExample(guid=guid, text_a=text_c, text_b=re_texta + re_textb,
                                     text_c=text_a + text_b,
                                     label="0"))
                else:
                    # corrupting tail
                    while True:
                        tmp_ent_list = set(entities)
                        tmp_ent_list.remove(test_triple[2])
                        tmp_ent_list = list(tmp_ent_list)
                        tmp_tail = random.choice(tmp_ent_list)
                        tmp_triple_str = test_triple[0] + '\t' + test_triple[1] + '\t' + tmp_tail
                        if tmp_triple_str not in lines_str_set:
                            break
                    tmp_tail_text = ent2text[tmp_tail]
                    text_b = tmp_tail_text
                    re_textb = ""
                    search_key = tmp_tail  # 可以替换为您想要搜索的键
                    # 使用之前解析的数据
                    contextlist = self.process_list_items(parsed_data, search_key, test_triple[1], rel2text)
                    re_textb += (" " + contextlist)
                    examples_negative.append(
                        InputExample(guid=guid, text_a=text_c, text_b=re_texta + re_textb,
                                     text_c=text_a + text_b,
                                     label="0"))
                examples_negative = list(set(examples_negative)) #去重
            examples.extend(examples_negative)
        return examples


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, print_info=False):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label: i for i, label in enumerate(label_list)}
    cout = 0;
    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0 and print_info:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        tokens_c = None

        if example.text_b and example.text_c:
            tokens_b = tokenizer.tokenize(example.text_b)
            tokens_c = tokenizer.tokenize(example.text_c)
            total_length = len(tokens_a) + len(tokens_b) + len(tokens_c)
            if total_length > max_seq_length:
                cout = cout + 1
            # print(cout)

            _truncate_seq_triple(tokens_a, tokens_b, tokens_c, max_seq_length - 4)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [0] * (len(tokens_b) + 1)
        if tokens_c:
            tokens += tokens_c + ["[SEP]"]
            segment_ids += [1] * (len(tokens_c) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_id = label_map[example.label]

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_id))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def _truncate_seq_triple(tokens_a, tokens_b, tokens_c, max_length):
    """Truncates a sequence triple in place to the maximum length."""

    while True:
        total_length = len(tokens_a) + len(tokens_b) + len(tokens_c)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b) and len(tokens_a) > len(tokens_c):
            tokens_a.pop()
        elif len(tokens_b) > len(tokens_a) and len(tokens_b) > len(tokens_c):
            tokens_b.pop()
        elif len(tokens_c) > len(tokens_a) and len(tokens_c) > len(tokens_b):
            tokens_c.pop()
        else:
            tokens_c.pop()


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
    if task_name == "kg":
        return {"acc": simple_accuracy(preds, labels)}
    else:
        raise KeyError(task_name)


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir",
                        default='sub_data/WN18RR_v1',
                        type=str,
                        required=False,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--bert_model", default='./cased', type=str, required=False,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                             "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--task_name",
                        default='KG',
                        type=str,
                        required=False,
                        help="The name of the task to train.")
    parser.add_argument("--output_dir",
                        default='output/hits/wn18rr_v1_4e-5-400',
                        type=str,
                        required=False,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parametersguazh
    parser.add_argument("--cache_dir",
                        default="",
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length",
                        default=400,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        default=True,
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        default=False,
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_predict",
                        default=True,
                        action='store_true',
                        help="Whether to run eval on the test set.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=64,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=4e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")
    args = parser.parse_args()

    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    processors = {
        "kg": KGProcessor,
    }

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)

    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps
    args.seed = random.randint(1, 200)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_predict:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    task_name = args.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()

    label_list = processor.get_labels(args.data_dir)
    num_labels = len(label_list)

    entity_list = processor.get_entities(args.data_dir)
    # print(entity_list)

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    train_examples = None
    num_train_optimization_steps = 0
    if args.do_train:
        train_examples = processor.get_train_examples(args.data_dir)
        num_train_optimization_steps = int(
            len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps + 1) * args.num_train_epochs
        if args.local_rank != -1:
            num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

    # Prepare model
    cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE),
                                                                   'distributed_{}'.format(args.local_rank))
    model = BertForSequenceClassification.from_pretrained(args.bert_model,
                                                          cache_dir=cache_dir,
                                                          num_labels=num_labels)
    # model = CustomBertForSequenceClassification(args.bert_model,
    #                                             cache_dir=cache_dir,
    #                                             num_labels=num_labels)
    if args.fp16:
        model.half()
    model.to(device)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    if args.fp16:
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              bias_correction=False,
                              max_grad_norm=1.0)
        if args.loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)
        warmup_linear = WarmupLinearSchedule(warmup=args.warmup_proportion,
                                             t_total=num_train_optimization_steps)

    else:
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=num_train_optimization_steps)

    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0
    if args.do_train:

        train_features = convert_examples_to_features(
            train_examples, label_list, args.max_seq_length, tokenizer)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)

        all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)

        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        model.train()
        # print(model)
        for _ in trange(int(args.num_train_epochs), desc="Epoch"):
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch

                # define a new function to compute loss values for both output_modes
                logits = model(input_ids, segment_ids, input_mask, labels=None)

                # print(logits, logits.shape)

                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))

                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if args.fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()

                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16:
                        # modify learning rate with special warm up BERT uses
                        # if args.fp16 is False, BertAdam is used that handles this automatically
                        lr_this_step = args.learning_rate * warmup_linear.get_lr(
                            global_step / num_train_optimization_steps,
                            args.warmup_proportion)
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr_this_step
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1
            print("Training loss: ", tr_loss, nb_tr_examples)

    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Save a trained model, configuration and tokenizer
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self

        # If we save using the predefined names, we can load using `from_pretrained`
        output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
        output_config_file = os.path.join(args.output_dir, CONFIG_NAME)

        torch.save(model_to_save.state_dict(), output_model_file)
        model_to_save.config.to_json_file(output_config_file)
        tokenizer.save_vocabulary(args.output_dir)

        # Load a trained model and vocabulary that you have fine-tuned
        model = BertForSequenceClassification.from_pretrained(args.output_dir, num_labels=num_labels)
        tokenizer = BertTokenizer.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
    else:
        model = BertForSequenceClassification.from_pretrained(args.bert_model, num_labels=num_labels)
    model.to(device)

    def get_test_entities(data_dir):

        with open(os.path.join(data_dir, "test.txt"), mode='r', encoding='utf-8') as file:
            tsv_reader = list(csv.reader(file, delimiter='\t'))

        # 初始化已访问元素集合
        visited_elements = set()
        # 提取已访问元素集合
        for row in tsv_reader:
            visited_elements.add(row[0])
            visited_elements.add(row[2])
        return visited_elements

    if args.do_predict and (args.local_rank == -1 or torch.distributed.get_rank() == 0):

        train_triples = processor.get_train_triples(args.data_dir)
        dev_triples = processor.get_dev_triples(args.data_dir)
        test_triples = processor.get_test_triples(args.data_dir)
        all_triples = train_triples + dev_triples + test_triples

        test_str_set = set()
        for triple in test_triples:
            triple_str = '\t'.join(triple)
            test_str_set.add(triple_str)

        # Load a trained model and vocabulary that you have fine-tuned
        model = BertForSequenceClassification.from_pretrained(args.output_dir, num_labels=num_labels)
        tokenizer = BertTokenizer.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        model.to(device)
        model.eval()

        ranks = []
        top_ten_hit_count = 0

        for test_triple in test_triples:
            # Create negative samples
            # Convert examples to features
            tmp_examples = processor.create_examples_negative(test_triple, "test", args.data_dir, test_str_set)
            tmp_features = convert_examples_to_features(tmp_examples, label_list, args.max_seq_length, tokenizer,
                                                        print_info=False)

            # Create dataloader
            eval_data = TensorDataset(
                torch.tensor([f.input_ids for f in tmp_features], dtype=torch.long),
                torch.tensor([f.input_mask for f in tmp_features], dtype=torch.long),
                torch.tensor([f.segment_ids for f in tmp_features], dtype=torch.long),
                torch.tensor([f.label_id for f in tmp_features], dtype=torch.long)
            )

            eval_dataloader = DataLoader(eval_data, sampler=SequentialSampler(eval_data),
                                         batch_size=args.eval_batch_size)

            preds = []
            for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader, desc="Testing"):
                input_ids = input_ids.to(device)
                input_mask = input_mask.to(device)
                segment_ids = segment_ids.to(device)

                with torch.no_grad():
                    logits = model(input_ids, segment_ids, input_mask, labels=None)

                if len(preds) == 0:
                    preds.append(logits.detach().cpu().numpy())
                else:
                    preds[0] = np.append(preds[0], logits.detach().cpu().numpy(), axis=0)

            preds = np.vstack(preds)
            rel_values = torch.tensor(preds[:, 1])
            #print("每个样本的第一个预测值 (rel_values):\n", rel_values)
            _, argsort1 = torch.sort(rel_values, descending=True)
            rank = np.where(argsort1.cpu().numpy() == 0)[0][0]
            ranks.append(rank + 1)

            if rank < 3:
                top_ten_hit_count += 1

            print(f'当前排名: {rank}, 平均排名: {np.mean(ranks)}, Hit@10: {top_ten_hit_count / len(ranks)}')

        # 输出最终结果
        logger.info(f'最终平均排名: {np.mean(ranks)}')
        logger.info(f'最终Hit@10: {top_ten_hit_count / len(ranks)}')


if __name__ == "__main__":
    main()
