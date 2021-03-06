import os
import sys

import numpy as np

from common import argument_parser
from common import load_model, load_tsv_data
from common import tokenize_texts, encode_tokenized


def main(argv):
    args = argument_parser('test').parse_args(argv[1:])

    model, tokenizer, labels, config = load_model(args.model_dir)
    test_labels, test_texts = load_tsv_data(args.test_data, args)

    max_seq_len = config['max_seq_length']
    replace_span = config['replace_span']

    label_map = { t: i for i, t in enumerate(labels) }
    inv_label_map = { v: k for k, v in label_map.items() }

    test_tok = tokenize_texts(test_texts, tokenizer)
    test_x = encode_tokenized(test_tok, tokenizer, max_seq_len, replace_span)
    test_y = [label_map[l] for l in test_labels]

    probs = model.predict(test_x, batch_size=args.batch_size)
    preds = np.argmax(probs, axis=-1)
    correct, total = sum(g==p for g, p in zip(test_y, preds)), len(test_y)
    print('Test accuracy: {:.1%} ({}/{})'.format(
        correct/total, correct, total))
    
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
