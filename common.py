#!/usr/bin/env python3

import sys
import os
import re
import json

import numpy as np
import tensorflow as tf

os.environ['TF_KERAS'] = '1'

from itertools import count
from functools import wraps
from time import time
from argparse import ArgumentParser
from logging import info, warning

from tensorflow import keras
import bert_tokenization as tokenization
from keras_bert import load_trained_model_from_checkpoint
from keras_bert import calc_train_steps, AdamWarmup
from keras_bert import get_custom_objects

from tensorflow.keras.layers import Average, Concatenate
from tensorflow.keras.utils import Sequence
from tensorflow.keras.callbacks import Callback

from config import DEFAULT_SEQ_LEN, DEFAULT_BATCH_SIZE, DEFAULT_EPOCHS
from config import DEFAULT_LR, DEFAULT_WARMUP_PROPORTION
from config import DEFAULT_MAX_CHECKPOINTS, CHECKPOINT_NAME


def print_versions(out=sys.stderr):
    print('Using tensorflow {}'.format(tf.__version__), file=sys.stderr)
    print('Using keras {}'.format(keras.__version__), file=sys.stderr)


def timed(f, out=sys.stderr):
    @wraps(f)
    def wrapper(*args, **kwargs):
        start = time()
        result = f(*args, **kwargs)
        print('@timed: {} completed in {:.1f} sec'.format(
            f.__name__, time()-start), file=out, flush=True)
        return result
    return wrapper


def argument_parser(mode):
    argparser = ArgumentParser()
    if mode == 'train':
        argparser.add_argument(
            '--train_data', required=True,
            help='Training data'
        )
        argparser.add_argument(
            '--labels', required=True,
            help='File containing list of labels'
        )
        argparser.add_argument(
            '--dev_data', default=None,
            help='Development data'
        )
        argparser.add_argument(
            '--task_name', default="NER",
            help='task to run, acceptable values NER and RE'
        )
        argparser.add_argument(
            '--vocab_file', required=True,
            help='Vocabulary file that BERT model was trained on'
        )
        argparser.add_argument(
            '--bert_config_file', required=True,
            help='Configuration for pre-trained BERT model'
        )
        argparser.add_argument(
            '--init_checkpoint', required=True,
            help='Initial checkpoint for pre-trained BERT model'
        )
        argparser.add_argument(
            '--max_seq_length', type=int, default=DEFAULT_SEQ_LEN,
            help='Maximum input sequence length in WordPieces'
        )
        argparser.add_argument(
            '--output-layer', default='-1',
            help='BERT output layer (int, -1 for last, "avg", or "concat")'
        )
        argparser.add_argument(
            '--do_lower_case', default=False, action='store_true',
            help='Lower case input text (for uncased models)'
        )
        argparser.add_argument(
            '--learning_rate', type=float, default=DEFAULT_LR,
            help='Initial learning rate'
        )
        argparser.add_argument(
            '--num_train_epochs', type=int, default=DEFAULT_EPOCHS,
            help='Number of training epochs'
        )
        argparser.add_argument(
            '--warmup_proportion', type=float, default=DEFAULT_WARMUP_PROPORTION,
            help='Proportion of training to perform LR warmup for'
        )
        argparser.add_argument(
            '--replace_span', default=None,
            help='Replace span text with given special token'
        )
        argparser.add_argument(
            '--replace_span_A', default=None,
            help='Replace span text with given special token for first entity in RE'
        )
        argparser.add_argument(
            '--replace_span_B', default=None,
            help='Replace span text with given special token for second entity in RE'
        )
        argparser.add_argument(
            '--checkpoint_dir', default='checkpoints',
            help='Directory for model checkpoints'
        )
        argparser.add_argument(
            '--checkpoint_steps', type=int, default=None,
            help='How often to save model checkpoints'
        )
        argparser.add_argument(
            '--max_checkpoints', type=int, default=DEFAULT_MAX_CHECKPOINTS,
            help='Maximum number of checkpoints to store'
        )
    argparser.add_argument(
        '--label_field', type=int, default=-4,
        help='Index of label in TSV data (1-based)'
    )
    argparser.add_argument(
        '--text_fields', type=int, default=-3,
        help='Index of first text field in TSV data (1-based)'
    )
    if mode != 'serve':
        test_data_required = mode in ('test', 'predict',)
        argparser.add_argument(
            '--test_data', required=test_data_required,
            help='Test data'
        )
    argparser.add_argument(
        '--batch_size', type=int, default=DEFAULT_BATCH_SIZE,
        help='Batch size for training'
    )
    model_dir_required = mode in ('test', 'predict', 'serve')
    argparser.add_argument(
        '--model_dir', default=None, required=model_dir_required,
        help='Trained model directory'
    )
    if mode == 'serve':
        argparser.add_argument(
            '--port', default=9000,
            help='Port to listen to'
        )
    return argparser


def get_checkpoint_files(directory, name=CHECKPOINT_NAME):
    filenames = []
    regex = re.compile(r'^' + re.sub(r'{.*}', r'.*', name) + r'$')
    for fn in os.listdir(directory):
        if regex.match(fn):
            filenames.append(fn)
    paths = [ os.path.join(directory, fn) for fn in filenames ]
    paths.sort(key=os.path.getctime, reverse=True)
    return paths


def delete_old_checkpoints(directory, name, max_checkpoints):
    paths = get_checkpoint_files(directory, name)
    delete = paths[max_checkpoints:]
    if delete:
        print('Deleting {}/{} checkpoints: {}'.format(
            len(delete), len(paths), delete), file=sys.stderr, flush=True)
    for path in delete:
        os.remove(path)


class DeleteOldCheckpoints(Callback):
    def __init__(self, checkpoint_dir, checkpoint_name, max_checkpoints):
        self._checkpoint_dir = checkpoint_dir
        self._checkpoint_name = checkpoint_name
        self._max_checkpoints = max_checkpoints

    def on_batch_end(self, batch, logs=None):
        delete_old_checkpoints(
            self._checkpoint_dir, self._checkpoint_name, self._max_checkpoints)


@timed
def load_pretrained(options):
    model = load_trained_model_from_checkpoint(
        options.bert_config_file,
        options.init_checkpoint,
        training=False,
        trainable=True,
        seq_len=options.max_seq_length,
    )
    return model


def get_tokenizer(options):
    tokenizer = tokenization.FullTokenizer(
        vocab_file=options.vocab_file,
        do_lower_case=options.do_lower_case
    )
    return tokenizer


def get_bert_output(model, layer_index, output_offset):
    if layer_index == -1:
        layer_output = model.output
    else:
        layer_name = 'Encoder-{}-FeedForward-Norm'.format(layer_index)
        layer_output = model.get_layer(layer_name).output
    return layer_output[:, output_offset]


def is_signed_digit(s):
    if type(s) == int:
        return True
    elif s.startswith('-'):
        return s[1:].isdigit()
    else:
        return s.isdigit()


def create_model(pretrained_model, num_labels, output_offset,
                 layer_index):
    model_inputs = pretrained_model.inputs[:2]
    if is_signed_digit(layer_index):
        layer_index = int(layer_index)
        pretrained_output = get_bert_output(pretrained_model, layer_index,
                                            output_offset)
    elif layer_index in ('avg', 'concat'):
        outputs = []
        for i in count(1):
            try:
                outputs.append(get_bert_output(pretrained_model, i,
                                               output_offset))
            except ValueError:
                break    # assume past last layer
        if layer_index == 'avg':
            pretrained_output = Average()(outputs)
        else:
            assert layer_index == 'concat'
            pretrained_output = Concatenate()(outputs)

    model_output = keras.layers.Dense(
        num_labels,
        activation='softmax'
    )(pretrained_output)
    model = keras.models.Model(inputs=model_inputs, outputs=model_output)
    return model


def _model_path(model_dir):
    return os.path.join(model_dir, 'model.hdf5')


def _vocab_path(model_dir):
    return os.path.join(model_dir, 'vocab.txt')


def _labels_path(model_dir):
    return os.path.join(model_dir, 'labels.txt')


def _config_path(model_dir):
    return os.path.join(model_dir, 'config.json')


def save_model_etc(model, tokenizer, labels, options):
    # TODO rename
    os.makedirs(options.model_dir, exist_ok=True)
    config = {
        'do_lower_case': options.do_lower_case,
        'max_seq_length': options.max_seq_length,
        'replace_span': options.replace_span,
    }
    with open(_config_path(options.model_dir), 'w') as out:
        json.dump(config, out, indent=4)
    model.save(_model_path(options.model_dir))
    with open(_labels_path(options.model_dir), 'w') as out:
        for label in labels:
            print(label, file=out)
    with open(_vocab_path(options.model_dir), 'w') as out:
        for i, v in sorted(list(tokenizer.inv_vocab.items())):
            print(v, file=out)


def load_model(model_path):
    return keras.models.load_model(
        model_path,
        custom_objects=get_custom_objects()
    )


def load_model_etc(model_dir):
    with open(_config_path(model_dir)) as f:
        config = json.load(f)
    model = load_model(_model_path(model_dir))
    tokenizer = tokenization.FullTokenizer(
        vocab_file=_vocab_path(model_dir),
        do_lower_case=config['do_lower_case']
    )
    labels = load_labels(_labels_path(model_dir))
    return model, tokenizer, labels, config


def load_labels(path):
    labels = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line in labels:
                raise ValueError('duplicate value {} in {}'.format(line, path))
            labels.append(line)
    return labels


def create_optimizer(num_example, batch_size, options):
    total_steps, warmup_steps = calc_train_steps(
        num_example=num_example,
        batch_size=batch_size,
        epochs=options.num_train_epochs,
        warmup_proportion=options.warmup_proportion,
    )
    print('optimizer total_steps: {}, warmup_steps: {}'.format(
        total_steps, warmup_steps), file=sys.stderr)
    optimizer = AdamWarmup(
        total_steps,
        warmup_steps,
        lr=options.learning_rate,
        epsilon=1e-6,
        weight_decay=0.01,
        weight_decay_pattern=['embeddings', 'kernel', 'W1', 'W2', 'Wk', 'Wq', 'Wv', 'Wo']
    )
    return optimizer

def fix_unused_tokens(tokenized_text):
    for i,v in enumerate(tokenized_text):
        if tokenized_text[i:i+2] == ["unused", "##3"]:
            tokenized_text.append("[unused3]")
            #remove the ##3 token from the list of tokens
            tokenized_text.pop(i+1)
        else:
            tokenized_text.append(tokenized_text[i])
    return tokenized_text

def tokenize_texts_re(texts, tokenizer):
    tokenized = []
    for sent_start, entity1, text_between_ent_1_and_ent_2, entity2, sent_end in texts:
        sent_start_tok = fix_unused_tokens(tokenizer.tokenize(sent_start))
        entity1_tok = fix_unused_tokens(tokenizer.tokenize(entity1))
        text_between_ent_1_and_ent_2_tok = fix_unused_tokens(tokenizer.tokenize(text_between_ent_1_and_ent_2))
        entity2_tok = fix_unused_tokens(tokenizer.tokenize(entity2))
        sent_end_tok = fix_unused_tokens(tokenizer.tokenize(sent_end))
        tokenized.append([sent_start_tok, entity1_tok, text_between_ent_1_and_ent_2_tok, entity2_tok, sent_end_tok])
    return tokenized

def tokenize_texts(texts, tokenizer):
    tokenized = []
    for left, span, right in texts:
        left_tok = tokenizer.tokenize(left)
        span_tok = tokenizer.tokenize(span)
        right_tok = tokenizer.tokenize(right)
        tokenized.append([left_tok, span_tok, right_tok])
    return tokenized

def encode_tokenized(tokenized_texts, tokenizer, seq_len, replace_span):
    tids, sids = [], []
    for left, span, right in tokenized_texts:
        tokens = ['[CLS]']
        center = int(seq_len/2)
        if len(left) > center-1:    # -1 for CLS
            left = left[len(left)-(center-1):]
        else:
            left = ['[PAD]'] * ((center-1)-len(left)) + left
        tokens.extend(left)
        if not replace_span:
            tokens.extend(span)
        else:
            tokens.append(replace_span)
        tokens.extend(right)
        if len(tokens) >= seq_len-1:    # -1 for [SEP]
            tokens, chopped = tokens[:seq_len-1], tokens[seq_len-1:]
            info('chopping tokens to {}: {} ///// {}'.format(
                seq_len-1, ' '.join(tokens), ' '.join(chopped)))
        tokens.append('[SEP]')
        tokens.extend(['[PAD]'] * (seq_len-len(tokens)))
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        segment_ids = [0] * seq_len
        tids.append(token_ids)
        sids.append(segment_ids)
    # Sanity check
    assert all(len(t) == seq_len for t in tids)
    assert all(len(s) == seq_len for s in sids)
    return np.array(tids), np.array(sids)

def encode_tokenized_re(tokenized_texts, tokenizer, seq_len, replace_span_A, replace_span_B):
    tids, sids = [], []
    for sent_start_tok, entity1_tok, text_between_ent_1_and_ent_2_tok, entity2_tok, sent_end_tok in tokenized_texts:
        tokens = ['[CLS]']
        center = int(seq_len/2)
        if (len(sent_start_tok+entity1_tok)+int(round(len(text_between_ent_1_and_ent_2_tok)/2))) > center-1:
            sent_start_tok = sent_start_tok[len(sent_start_tok+entity1_tok)+int(round(len(text_between_ent_1_and_ent_2_tok)/2))-(center-1):]
        else:
            sent_start_tok = ['[PAD]'] * ((center-1)-len(sent_start_tok+entity1_tok)+int(round(len(text_between_ent_1_and_ent_2_tok)/2))) + sent_start_tok
        tokens.extend(sent_start_tok)

        if not replace_span_A:
            tokens.extend(entity1_tok)
        else:
            tokens.append(replace_span_A)
        tokens.extend(text_between_ent_1_and_ent_2_tok)

        if not replace_span_B:
            tokens.extend(entity2_tok)
        else:
            tokens.append(replace_span_B)
        tokens.extend(sent_end_tok)

        if len(tokens) >= seq_len -1:
            tokens, chopped = tokens[:seq_len-1], tokens[seq_len-1:]
            #shows the chopped inputs, log files for 10M end up being 3gb because of that so I stopped logging that
            #logging.warning('chopping tokens to {}: {} ///// {}'.format(max_seq_length-1, ' '.join(tokens), ' '.join(chopped)))
        tokens.append('[SEP]')
        tokens.extend(['[PAD]'] * (seq_len-len(tokens)))
        segment_ids = []
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = []
        for token in tokens:
            if token == "[PAD]":
                input_mask.append(0)
            else:
                input_mask.append(1)
        segment_ids = [0] * seq_len
        tids.append(token_ids)
        sids.append(segment_ids)
    # Sanity check
    assert all(len(t) == seq_len for t in tids)
    assert all(len(s) == seq_len for s in sids)
    return np.array(tids), np.array(sids)


def positive_index(i, fields):
    return i if i >= 0 else len(fields)+i


def parse_tsv_line(l, ln, fn, options):
    l = l.rstrip('\n')
    fields = l.split('\t')
    if len(fields) < 4:
        raise ValueError(
            'Expected at least 4 tab-separated fields, got '
            '{} on {} line {}: {}'.format(len(fields), fn, ln, l)
        )
    label = fields[options.label_field]
    text_end = positive_index(options.text_fields, fields) + 3
    text = fields[options.text_fields:text_end]
    return label, text


def load_tsv_data(fn, options):
    labels, texts = [], []
    with open(fn) as f:
        for ln, l in enumerate(f, start=1):
            label, text = parse_tsv_line(l, ln, fn, options)
            labels.append(label)
            texts.append(text)
    return labels, texts


def encode_data(texts, labels, tokenizer, max_seq_len, replace_span, label_map,
                options):
    tokenized = tokenize_texts(texts, tokenizer)
    x = encode_tokenized(tokenized, tokenizer, max_seq_len, replace_span)
    y = np.array([label_map[l] for l in labels])
    return x, y

def encode_data_re(texts, labels, tokenizer, max_seq_len, replace_span_A, replace_span_B, label_map,
                options):
    tokenized = tokenize_texts_re(texts, tokenizer)
    x = encode_tokenized_re(tokenized, tokenizer, max_seq_len, replace_span_A, replace_span_B)
    y = np.array([label_map[l] for l in labels])
    return x, y

@timed
def load_dataset(fn, tokenizer, max_seq_len, replace_span, label_map, options):
    labels, texts = load_tsv_data(fn, options)
    return encode_data(texts, labels, tokenizer, max_seq_len, replace_span,
                       label_map, options)

@timed
def load_dataset_re(fn, tokenizer, max_seq_len, replace_span_A, replace_span_B, label_map, options):
    labels, texts = load_tsv_data(fn, options)
    return encode_data_re(texts, labels, tokenizer, max_seq_len, replace_span_A, replace_span_B,
                       label_map, options)

@timed
def load_batch_offsets(fn, batch_size):
    offsets, offset = [], 0
    with open(fn, 'rb') as f:
        for ln, l in enumerate(f):
            if ln % batch_size == 0:
                offsets.append(offset)
            offset += len(l)
    return offsets, ln


def load_batch_from_tsv(fn, base_ln, offset, batch_size, options,
                        encoding='utf-8'):
    labels, texts = [], []
    with open(fn, 'rb') as f:
        f.seek(offset)
        for ln, l in enumerate(f):
            if len(texts) >= batch_size:
                break
            l = l.decode(encoding)
            label, text = parse_tsv_line(l, base_ln+ln, fn, options)
            labels.append(label)
            texts.append(text)
    return labels, texts


def tsv_generator(data_path, tokenizer, label_map, options):
    max_seq_len = options.max_seq_length
    replace_span = options.replace_span
    with open(data_path) as f:
        for ln, l in enumerate(f, start=1):
            label, text = parse_tsv_line(l, ln, data_path, options)
            # TODO function to encode single example
            (t, s), y = encode_data([text], [label], tokenizer, max_seq_len,
                                    replace_span, label_map, options)
            yield (t[0], s[0]), y[0]


def num_tsv_examples(fn):
    return sum(1 for _ in open(fn))


def num_tfrecord_examples(fn):
    return sum(1 for _ in tf.data.TFRecordDataset(fn))


@timed
def num_examples(fn):
    if isinstance(fn, list):
        return sum(num_examples(f) for f in fn)
    elif fn.endswith('.tsv'):
        return num_tsv_examples(fn)
    elif fn.endswith('.tfrecord'):
        return num_tfrecord_examples(fn)
    else:
        raise ValueError('file {} must be .tsv or .tfrecord'.format(fn))


def get_decode_function(max_seq_len):
    name_to_features = {
        'Input-Token': tf.io.FixedLenFeature([max_seq_len], tf.int64),
        'Input-Segment': tf.io.FixedLenFeature([max_seq_len], tf.int64),
        'label': tf.io.FixedLenFeature([1], tf.int64),
    }
    def decode_tfrecord(record):
        example = tf.io.parse_single_example(record, name_to_features)
        t = tf.cast(example['Input-Token'], tf.int32)
        s = tf.cast(example['Input-Segment'], tf.int32)
        y = tf.cast(example['label'], tf.int32)
        x = (t, s)
        return x, y
    return decode_tfrecord


def train_tfrecord_input(filenames, max_seq_len, batch_size, num_threads=10):
    # Largely following BERT run_pretraining.py with is_training=True,
    # including shuffling and parallel reading.
    dataset = tf.data.Dataset.from_tensor_slices(filenames)
    dataset = dataset.repeat().shuffle(buffer_size=len(filenames))
    max_concurrent = min(num_threads, len(filenames))
    dataset = dataset.interleave(
        tf.data.TFRecordDataset,
        cycle_length=max_concurrent,
        num_parallel_calls=max_concurrent
    )
    decode = get_decode_function(max_seq_len)
    dataset = dataset.map(decode, num_parallel_calls=num_threads)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(1)    # TODO optimize
    return dataset


def load_tfrecords(fn, max_seq_len, batch_size):
    decode = get_decode_function(max_seq_len)
    # TODO support multiple TFRecords
    dataset = tf.data.TFRecordDataset(fn)
    dataset = dataset.map(decode, num_parallel_calls=10)    # TODO
    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(1)    # TODO optimize
    return dataset


class TsvSequence(Sequence):
    def __init__(self, data_path, tokenizer, label_map, batch_size, options):
        self._data_path = data_path
        self._tokenizer = tokenizer
        self._label_map = label_map
        self._batch_size = batch_size
        self._max_seq_len = options.max_seq_length
        self._replace_span = options.replace_span
        self._options = options
        offsets, total = load_batch_offsets(data_path, batch_size)
        self._batch_offsets = offsets
        self.num_examples = total

    def __len__(self):
        return len(self._batch_offsets)

    def __getitem__(self, idx):
        base_ln = idx * self._batch_size
        offset = self._batch_offsets[idx]
        labels, texts = load_batch_from_tsv(self._data_path, base_ln, offset,
                                            self._batch_size, self._options)
        x, y = encode_data(texts, labels, self._tokenizer, self._max_seq_len,
                           self._replace_span, self._label_map, self._options)
        return x, y

    def __on_epoch_end__(self):
        pass

class TsvSequenceRE(Sequence):
    def __init__(self, data_path, tokenizer, label_map, batch_size, options):
        self._data_path = data_path
        self._tokenizer = tokenizer
        self._label_map = label_map
        self._batch_size = batch_size
        self._max_seq_len = options.max_seq_length
        self._replace_span_A = options.replace_span_A
        self._replace_span_B = options.replace_span_B
        self._options = options
        offsets, total = load_batch_offsets(data_path, batch_size)
        self._batch_offsets = offsets
        self.num_examples = total

    def __len__(self):
        return len(self._batch_offsets)

    def __getitem__(self, idx):
        base_ln = idx * self._batch_size
        offset = self._batch_offsets[idx]
        labels, texts = load_batch_from_tsv(self._data_path, base_ln, offset,
                                            self._batch_size, self._options)
        x, y = encode_data_re(texts, labels, self._tokenizer, self._max_seq_len,
                           self._replace_span_A, self._replace_span_B, self._label_map, self._options)
        return x, y

    def __on_epoch_end__(self):
        pass
