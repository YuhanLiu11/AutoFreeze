
import csv
import os
import logging
import argparse
import random
from tqdm import tqdm, trange
import pandas as pd
import math
import numpy as np
import torch
import tokenization
class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

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

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        print(quotechar)
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines


class ColaProcessor(DataProcessor):
    """Processor for the CoLA data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            None,
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir, data_num=None):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        test_mode = set_type == "test"
        
        if test_mode:
            lines = lines[1:]
        text_index = 1 if test_mode else 3
        examples = []
        for (i, line) in enumerate(lines):
            
            guid = "%s-%s" % (set_type, i)
            text_a = line[text_index]
            label = None if test_mode else line[1]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples
        
class AGNewsProcessor(DataProcessor):
    """Processor for the AG data set."""

    def get_train_examples(self, data_dir, data_num=None):
        """See base class."""
        train_data = pd.read_csv(os.path.join(data_dir, "train.csv"),header=None).values
        return self._create_examples(train_data, "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        dev_data = pd.read_csv(os.path.join(data_dir, "test.csv"),header=None).values
        return self._create_examples(dev_data, "dev")

    def get_labels(self):
        """See base class."""
        return ["1","2","3","4"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):	
            # if i > 600: break
            guid = "%s-%s" % (set_type, i)
            text_a = tokenization.convert_to_unicode(line[1]+" - "+line[2])
            label = tokenization.convert_to_unicode(str(line[0]))
            if i%1000==0:
                print(i)
                print("guid=",guid)
                print("text_a=",text_a)
                print("label=",label)
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class AGNewsProcessor_sep(DataProcessor):
    """Processor for the AG data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        train_data = pd.read_csv(os.path.join(data_dir, "train.csv"),header=None).values
        return self._create_examples(train_data, "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        dev_data = pd.read_csv(os.path.join(data_dir, "test.csv"),header=None).values
        return self._create_examples(dev_data, "dev")

    def get_labels(self):
        """See base class."""
        return ["1","2","3","4"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):	
            guid = "%s-%s" % (set_type, i)
            text_a = tokenization.convert_to_unicode(line[1])
            text_b = tokenization.convert_to_unicode(line[2])
            label = tokenization.convert_to_unicode(str(line[0]))
            if i%1000==0:
                print(i)
                print("guid=",guid)
                print("text_a=",text_a)
                print("text_b=",text_b)
                print("label=",label)
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

class AGNewsProcessor_sep_aug(DataProcessor):
    """Processor for the AG data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        train_data = pd.read_csv(os.path.join(data_dir, "train.csv"),header=None).values
        return self._create_examples(train_data, "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        dev_data = pd.read_csv(os.path.join(data_dir, "test.csv"),header=None).values
        return self._create_examples(dev_data, "dev")

    def get_labels(self):
        """See base class."""
        return ["1","2","3","4"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        if set_type=="train":
            scale=5
        else:
            scale=1
        examples = []
        for (i, line) in enumerate(lines):
            s=(line[1]+" - "+line[2]).split()
            l=len(s)
            for j in range(scale):
                r=random.randint(1,l-1)
                guid = "%s-%s" % (set_type, i*scale+j)
                text_a = tokenization.convert_to_unicode(" ".join(s[:r]))
                text_b = tokenization.convert_to_unicode(" ".join(s[r:]))
                label = tokenization.convert_to_unicode(str(line[0]))
                if i%1000==0:
                    print(i)
                    print("guid=",guid)
                    print("text_a=",text_a)
                    print("text_b=",text_b)
                    print("label=",label)
                examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

class IMDBProcessor(DataProcessor):
    """Processor for the IMDB data set."""

    def get_train_examples(self, data_dir, data_num=None):
        """See base class."""
        train_data = pd.read_csv(os.path.join(data_dir, "train.csv"),header=None,sep="\t").values
        return self._create_examples(train_data, "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        dev_data = pd.read_csv(os.path.join(data_dir, "test.csv"),header=None,sep="\t").values
        return self._create_examples(dev_data, "dev")

    def get_labels(self):
        """See base class."""
        return ["0","1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            # if i > 1199: break
            guid = "%s-%s" % (set_type, i)
            text_a = tokenization.convert_to_unicode(str(line[1]))
            label = tokenization.convert_to_unicode(str(line[0]))
            if i%1000==0:
                print(i)
                print("guid=",guid)
                print("text_a=",text_a)
                print("label=",label)
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class IMDBProcessor_sep(DataProcessor):
    """Processor for the IMDB data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        train_data = pd.read_csv(os.path.join(data_dir, "train.csv"),header=None,sep="\t").values
        return self._create_examples(train_data, "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        dev_data = pd.read_csv(os.path.join(data_dir, "test.csv"),header=None,sep="\t").values
        return self._create_examples(dev_data, "dev")

    def get_labels(self):
        """See base class."""
        return ["0","1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        scale=1
        examples = []
        for (i, line) in enumerate(lines):
            s=(line[1]).split()
            l=len(s)
            for j in range(scale):
                r=random.randint(1,l-1)
                guid = "%s-%s" % (set_type, i*scale+j)
                text_a = tokenization.convert_to_unicode(" ".join(s[:r]))
                text_b = tokenization.convert_to_unicode(" ".join(s[r:]))
                label = tokenization.convert_to_unicode(str(line[0]))
                if i%1000==0:
                    print(i)
                    print("guid=",guid)
                    print("text_a=",text_a)
                    print("text_b=",text_b)
                    print("label=",label)
                examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class IMDBProcessor_sep_aug(DataProcessor):
    """Processor for the IMDB data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        train_data = pd.read_csv(os.path.join(data_dir, "train.csv"),header=None,sep="\t").values
        return self._create_examples(train_data, "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        dev_data = pd.read_csv(os.path.join(data_dir, "test.csv"),header=None,sep="\t").values
        return self._create_examples(dev_data, "dev")

    def get_labels(self):
        """See base class."""
        return ["0","1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        if set_type=="train":
            scale=5
        else:
            scale=1
        examples = []
        for (i, line) in enumerate(lines):
            if i==100 and set_type=="train":break
            if i==1009 and set_type=="dev":break
            s=(line[1]).split()
            l=len(s)
            for j in range(scale):
                r=random.randint(1,l-1)
                guid = "%s-%s" % (set_type, i*scale+j)
                text_a = tokenization.convert_to_unicode(" ".join(s[:r]))
                text_b = tokenization.convert_to_unicode(" ".join(s[r:]))
                label = tokenization.convert_to_unicode(str(line[0]))
                if i%1000==0:
                    print(i)
                    print("guid=",guid)
                    print("text_a=",text_a)
                    print("text_b=",text_b)
                    print("label=",label)
                examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class Yelp_p_Processor(DataProcessor):
    """Processor for the IMDB data set."""

    def get_train_examples(self, data_dir, data_num=None):
        """See base class."""
        train_data = pd.read_csv(os.path.join(data_dir, "train.csv"),header=None,sep=",").values
        return self._create_examples(train_data, "train", data_num=data_num)

    def get_dev_examples(self, data_dir):
        """See base class."""
        dev_data = pd.read_csv(os.path.join(data_dir, "test.csv"),header=None,sep=",").values
        return self._create_examples(dev_data, "dev")

    def get_labels(self):
        """See base class."""
        return ["1","2"]

    def _create_examples(self, lines, set_type, data_num=None):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if data_num is not None and i > data_num and set_type == "train":
                break
            guid = "%s-%s" % (set_type, i)
            text_a = tokenization.convert_to_unicode(str(line[1]))
            label = tokenization.convert_to_unicode(str(line[0]))
            if i%1000==0:
                print(i)
                print("guid=",guid)
                print("text_a=",text_a)
                print("label=",label)
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

class Yelp_f_Processor(DataProcessor):
    """Processor for the IMDB data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        train_data = pd.read_csv(os.path.join(data_dir, "train.csv"),header=None,sep=",").values
        return self._create_examples(train_data, "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        dev_data = pd.read_csv(os.path.join(data_dir, "test.csv"),header=None,sep=",").values
        return self._create_examples(dev_data, "dev")

    def get_labels(self):
        """See base class."""
        return ["1","2","3","4","5"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            #if i>147:break
            guid = "%s-%s" % (set_type, i)
            text_a = tokenization.convert_to_unicode(str(line[1]))
            label = tokenization.convert_to_unicode(str(line[0]))
            if i%1000==0:
                print(i)
                print("guid=",guid)
                print("text_a=",text_a)
                print("label=",label)
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

        
class Yahoo_Processor(DataProcessor):
    """Processor for the IMDB data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        train_data = pd.read_csv(os.path.join(data_dir, "train.csv"),header=None,sep=",").values
        return self._create_examples(train_data, "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        dev_data = pd.read_csv(os.path.join(data_dir, "test.csv"),header=None,sep=",").values
        return self._create_examples(dev_data, "dev")

    def get_labels(self):
        """See base class."""
        return ["1","2","3","4","5","6","7","8","9","10"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            #if i>147:break
            guid = "%s-%s" % (set_type, i)
            text_a = tokenization.convert_to_unicode(str(line[1])+" "+str(line[2]))
            text_b = tokenization.convert_to_unicode(str(line[3]))
            label = tokenization.convert_to_unicode(str(line[0]))
            if i%1000==0:
                print(i)
                print("guid=",guid)
                print("text_a=",text_a)
                print("text_b=",text_b)
                print("label=",label)
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

class Trec_Processor(DataProcessor):
    """Processor for the IMDB data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        train_data = pd.read_csv(os.path.join(data_dir, "train.csv"),header=None,sep="\t").values
        return self._create_examples(train_data, "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        dev_data = pd.read_csv(os.path.join(data_dir, "test.csv"),header=None,sep="\t").values
        return self._create_examples(dev_data, "dev")

    def get_labels(self):
        """See base class."""
        return ['LOC', 'NUM', 'HUM', 'ENTY', 'ABBR', 'DESC']

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            #if i>147:break
            guid = "%s-%s" % (set_type, i)
            text_a = tokenization.convert_to_unicode(str(line[1]))
            label = tokenization.convert_to_unicode(str(line[0]))
            if i%1000==0:
                print(i)
                print("guid=",guid)
                print("text_a=",text_a)
                print("label=",label)
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

class Dbpedia_Processor(DataProcessor):
    """Processor for the IMDB data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        train_data = pd.read_csv(os.path.join(data_dir, "train.csv"),header=None,sep=",").values
        return self._create_examples(train_data, "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        dev_data = pd.read_csv(os.path.join(data_dir, "test.csv"),header=None,sep=",").values
        return self._create_examples(dev_data, "dev")

    def get_labels(self):
        """See base class."""
        return ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14']

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = tokenization.convert_to_unicode(str(line[1])+" "+str(line[2]))
            label = tokenization.convert_to_unicode(str(line[0]))
            if i%1000==0:
                print(i)
                print("guid=",guid)
                print("text_a=",text_a)
                print("label=",label)
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

class Sogou_Processor(DataProcessor):
    """Processor for the IMDB data set."""

    def get_train_examples(self, data_dir, data_num=None):
        """See base class."""
        train_data = pd.read_csv(os.path.join(data_dir, "train.csv"),header=None,sep="\t", encoding='utf-8-sig').values
        return self._create_examples(train_data, "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        dev_data = pd.read_csv(os.path.join(data_dir, "test.csv"),header=None,sep="\t",encoding='utf-8-sig').values
        return self._create_examples(dev_data, "dev")

    def get_labels(self):
        """See base class."""
        return ["1","2","3","4","5","6"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = tokenization.convert_to_unicode(str(line[1])+" "+str(line[2]))
            label = tokenization.convert_to_unicode(str(line[0]))
            if i%1000==0:
                print(i)
                print("guid=",guid)
                print("text_a=",text_a)
                print("label=",label)
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class MrpcProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def get_train_examples(self, data_dir,data_num=None):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "shuffled_train.tsv")))
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "shuffled_train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            if i > 500: break
            guid = "%s-%s" % (set_type, i)
            text_a = tokenization.convert_to_unicode(line[3])
            text_b = tokenization.convert_to_unicode(line[4])
            label = tokenization.convert_to_unicode(line[0])
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)

        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[0:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                segment_ids.append(1)
            tokens.append("[SEP]")
            segment_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_id = label_map[example.label]
        

        features.append(
                InputFeatures(
                        input_ids=input_ids,
                        input_mask=input_mask,
                        segment_ids=segment_ids,
                        label_id=label_id))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()
