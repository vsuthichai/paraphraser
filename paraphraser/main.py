import argparse
import tensorflow as tf
import numpy as np
import os
import sys
import datetime as dt
from six.moves import xrange
from lstm_model import lstm_model
from load_sent_embeddings import load_sentence_embeddings
from dataset_generator import ParaphraseDataset
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction

def optimizer(lr, loss):
    return tf.train.AdamOptimizer(lr).minimize(loss)

def debug_data(seq_source_ids, seq_ref_ids, seq_source_len, seq_ref_len, id_to_vocab):
    print("==============================================================")
    print("SOURCE!")
    print(seq_source_ids)
    for source_ids in seq_source_ids:
        print(' '.join([id_to_vocab[source_id] for source_id in source_ids]))
    print(seq_source_len)
    print("REFERENCE!")
    print(seq_ref_ids)
    for i in seq_ref_ids:
        print(' '.join([id_to_vocab[label] for label in i if label != -1]))
    print(seq_ref_len)
    print("==============================================================")
    return

def summarize_scalar(writer, tag, value, step):
    summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
    writer.add_summary(summary, step)

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir', type=str, default="logs", help="Log directory to store tensorboard summary and model checkpoints")
    parser.add_argument('--epochs', type=int, default=1, help="Number of epochs to train")
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate")
    parser.add_argument('--batch_size', type=int, default=64, help="Mini batch size")
    parser.add_argument('--max_seq_length', type=int, default=40, help="Maximum sequence length.  Sentence lengths beyond this are truncated.")
    parser.add_argument('--hidden_size', type=int, default=300, help="Hidden dimension size")

    return parser.parse_args()


def evaluate_validation_set(model, dataset_generator):
    pass



def main():
    args = parse_arguments()
    word_to_id, id_to_vocab, embeddings, start_id, end_id, unk_id = load_sentence_embeddings()
    mask_id = 5800
    vocab_size, embedding_size = embeddings.shape

    dataset = [
        { 
            'maxlen': 5,
            'train': '/media/sdb/datasets/aggregate_paraphrase_corpus_0/dataset.train.5',
            'dev': '/media/sdb/datasets/aggregate_paraphrase_corpus_0/dataset.dev.5',
            'test': '/media/sdb/datasets/aggregate_paraphrase_corpus_0/dataset.test.5' 
        },
        { 
            'maxlen': 10,
            'train': '/media/sdb/datasets/aggregate_paraphrase_corpus_0/dataset.train.10',
            'dev': '/media/sdb/datasets/aggregate_paraphrase_corpus_0/dataset.dev.10',
            'test': '/media/sdb/datasets/aggregate_paraphrase_corpus_0/dataset.test.10' 
        },
        { 
            'maxlen': 20,
            'train': '/media/sdb/datasets/aggregate_paraphrase_corpus_0/dataset.train.20',
            'dev': '/media/sdb/datasets/aggregate_paraphrase_corpus_0/dataset.dev.20',
            'test': '/media/sdb/datasets/aggregate_paraphrase_corpus_0/dataset.test.20' 
        },
        { 
            'maxlen': 30,
            'train': '/media/sdb/datasets/aggregate_paraphrase_corpus_0/dataset.train.30',
            'dev': '/media/sdb/datasets/aggregate_paraphrase_corpus_0/dataset.dev.30',
            'test': '/media/sdb/datasets/aggregate_paraphrase_corpus_0/dataset.test.30' 
        },
        { 
            'maxlen': 40,
            'train': '/media/sdb/datasets/aggregate_paraphrase_corpus_0/dataset.train.40',
            'dev': '/media/sdb/datasets/aggregate_paraphrase_corpus_0/dataset.dev.40',
            'test': '/media/sdb/datasets/aggregate_paraphrase_corpus_0/dataset.test.40' 
        },
        { 
            'maxlen': 50,
            'train': '/media/sdb/datasets/aggregate_paraphrase_corpus_0/dataset.train.50',
            'dev': '/media/sdb/datasets/aggregate_paraphrase_corpus_0/dataset.dev.50',
            'test': '/media/sdb/datasets/aggregate_paraphrase_corpus_0/dataset.test.50' 
        }
    ]

    dataset_generator = ParaphraseDataset(dataset, embeddings, word_to_id, start_id, end_id, unk_id, mask_id)
    start = dt.datetime.now()
    logdir = os.path.join(args.log_dir, "train-" + start.strftime("%Y%m%d-%H%M%S"))

    with tf.Session() as sess:
        model = lstm_model(args, embeddings, mask_id)
        train_step = optimizer(args.lr, model['loss'])
        writer = tf.summary.FileWriter(logdir)
        saver = tf.train.Saver()
        tf.global_variables_initializer().run()
        chencherry = SmoothingFunction()

        global_step = 0

        for epoch in xrange(args.epochs):
            generator = dataset_generator.generate_batch(args.batch_size, 'train', 10)
            #d = next(generator)
            #while 1:
            for d in generator:
                seq_source_ids = d['seq_source_ids']
                seq_source_words = d['seq_source_words']
                seq_source_len = d['seq_source_len']
                seq_ref_ids = d['seq_ref_ids']
                seq_ref_words = d['seq_ref_words']
                seq_ref_len = d['seq_ref_len']

                max_seq_source_len = max([len(source_ids) for source_ids in seq_source_ids ])
                max_seq_ref_len = max([len(ref_ids) for ref_ids in seq_ref_ids ])

                #debug_data(seq_source_ids, seq_ref_ids, seq_source_len, seq_ref_len, id_to_vocab)
                #return

                feed_dict = {
                    model['max_seq_source_len']: max_seq_source_len,
                    model['seq_source_ids']: seq_source_ids,
                    model['seq_source_lengths']: seq_source_len,

                    model['max_seq_ref_len']: max_seq_ref_len,
                    model['seq_reference_ids']: seq_ref_ids,
                    model['seq_reference_lengths']: seq_ref_len
                }

                feeds = [
                    train_step, 
                    model['loss'], 
                    model['predictions'], 
                    model['labels'], 
                    model['logits'], 
                    model['summaries'],
                    model['final_sequence_lengths']
                ]

                try:
                    _, batch_loss, predictions, labels, logits, summary, fsl = sess.run(feeds, feed_dict)
                except Exception as e:
                    debug_data(seq_source_ids, seq_ref_ids, seq_source_len, seq_ref_len, id_to_vocab)
                    raise e

                if global_step % 25 == 0:
                    writer.add_summary(summary, global_step)
                    writer.flush()
                    seq_ref_words = [ [ref_words] for ref_words in seq_ref_words ]
                    bleu_pred_words = [ [ id_to_vocab[vocab_id] for vocab_id in prediction if vocab_id in id_to_vocab ] for prediction in predictions ]
                    bleu_pred_words = [ pred_words[:pred_words.index('<END>') if '<END>' in pred_words else len(pred_words) ] for pred_words in bleu_pred_words ]
                    bleu_score = corpus_bleu(seq_ref_words, bleu_pred_words, smoothing_function=chencherry.method1)
                    summarize_scalar(writer, 'bleu_score', bleu_score, global_step)
                    print("{} : step={} epoch={} batch_loss={:.4f} bleu={:.4f}".format(dt.datetime.now(), global_step, epoch, batch_loss, bleu_score), flush=True)

                if global_step % 100 == 0:
                    evaluate_validation_set(model, dataset_generator)

                if global_step % 100 == 0:
                    debug_data(seq_source_ids, seq_ref_ids, seq_source_len, seq_ref_len, id_to_vocab)
                    print("PREDICTIONS!")
                    print("logits shape: " + str(logits.shape))
                    print("final_seq_lengths: " + str(fsl))
                    print("len(predictions): " + str(len(predictions)))
                    print("predictions: " + str(predictions))
                    for prediction in predictions:
                        print(str(len(prediction)) + ' ' + ' '.join([id_to_vocab[vocab_id] for vocab_id in prediction if vocab_id in id_to_vocab]))
                    print(predictions)
                
                if global_step % 5000 == 0 and global_step != 0:
                    saver.save(sess, os.path.join(logdir, 'model'), global_step=epoch)

                global_step += 1

if __name__ == '__main__':
    main()

