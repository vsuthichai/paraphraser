import argparse
import tensorflow as tf
import numpy as np
import os
import sys
import datetime as dt
from six.moves import xrange
from lstm_model import lstm_model
from load_dataset import generate_batch
from load_word_embeddings import load_glove_pickle

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir', type=str, help="Log directory to store tensorboard summary and model checkpoints")
    parser.add_argument('--epochs', type=int, default=2000, help="Number of epochs to train")
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate")
    parser.add_argument('--batch_size', type=int, default=64, help="Mini batch size")
    parser.add_argument('--max_seq_length', type=int, default=30, help="Maximum sequence length.  Sentence lengths beyond this are truncated.")
    parser.add_argument('--hidden_size', type=int, default=256, help="Hidden dimension size")
    parser.add_argument('--vocab_size', type=int, default=400000, help="Vocabulary size")

    return parser.parse_args()

def optimizer(lr, loss):
    return tf.train.AdamOptimizer(lr).minimize(loss)

def main():
    args = parse_arguments()

    with tf.Session() as sess:
        embeddings_filename = '/media/sdb/datasets/glove.6B/glove.6B.300d.pickle'
        vocab_to_id, id_to_vocab, np_embeddings = load_glove_pickle(embeddings_filename)
        model = lstm_model(args, np_embeddings)
        train_step = optimizer(args.lr, model['loss'])
        tf.global_variables_initializer().run()

        global_step = 0

        for epoch in xrange(args.epochs):
            for seq_source_ids, seq_ref_ids, seq_out_ids, seq_source_lengths, seq_ref_lengths in generate_batch(args.batch_size, args.max_seq_length, vocab_to_id, np_embeddings):

                #print(seq_source_ids)
                #for i in seq_source_ids:
                    #print(' '.join([id_to_vocab[label] for label in i if label != -1]))
                #print(seq_ref_ids)
                #print(seq_out_ids)
                #print(seq_ref_lengths)

                feed_dict = {
                    model['seq_source_ids']: seq_source_ids,
                    model['seq_reference_ids']: seq_ref_ids,
                    model['seq_output_ids']: seq_out_ids,
                    model['seq_source_lengths']: seq_source_lengths,
                    model['seq_reference_lengths']: seq_ref_lengths
                }

                _, loss, sample_id, labels, logits, embedding_source, encoder_state = sess.run([train_step, 
                                                                                     model['loss'], 
                                                                                     model['sample_id'],
                                                                                     model['labels'], 
                                                                                     model['logits'], 
                                                                                     model['embedding_source'], 
                                                                                     model['encoder_states']], feed_dict)
                print(loss)
                print(labels)
                print(logits.shape)
                for i in range(labels.shape[0]):
                    print(' '.join([id_to_vocab[label] for label in labels[i] if label != -1]))
                print(sample_id)
                for i in range(sample_id.shape[0]):
                    print(' '.join([id_to_vocab[label] for label in sample_id[i]]))

if __name__ == '__main__':
    main()

