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

def debug_data(seq_source_ids, seq_ref_ids, seq_source_len, seq_ref_len, id_to_vocab):
    print("==============================================================")
    print("SOURCE!")
    #print(seq_source_ids)
    for source_ids in seq_source_ids:
        print(' '.join([id_to_vocab[source_id] for source_id in source_ids]))
    print(seq_source_len)
    print("REFERENCE!")
    #print(seq_ref_ids)
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
    parser.add_argument('--epochs', type=int, default=2, help="Number of epochs to train")
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate")
    parser.add_argument('--batch_size', type=int, default=64, help="Mini batch size")
    parser.add_argument('--max_seq_length', type=int, default=40, help="Maximum sequence length.  Sentence lengths beyond this are truncated.")
    parser.add_argument('--hidden_size', type=int, default=300, help="Hidden dimension size")

    return parser.parse_args()

def main():
    args = parse_arguments()
    word_to_id, id_to_vocab, embeddings, start_id, end_id, unk_id = load_sentence_embeddings()
    mask_id = 5800
    vocab_size, embedding_size = embeddings.shape
    lr = args.lr

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
    train_logdir = os.path.join(args.log_dir, "train-" + start.strftime("%Y%m%d-%H%M%S"))
    dev_logdir = os.path.join(args.log_dir, "dev-" + start.strftime("%Y%m%d-%H%M%S"))

    with tf.Session() as sess:
        model = lstm_model(args, embeddings, mask_id)
        train_writer = tf.summary.FileWriter(train_logdir)
        dev_writer = tf.summary.FileWriter(dev_logdir)
        saver = tf.train.Saver()
        tf.global_variables_initializer().run()
        chencherry = SmoothingFunction()

        global_step = 0

        for epoch in xrange(args.epochs):
            train_losses = []
            train_batch_generator = dataset_generator.generate_batch(args.batch_size, 'train')
            #d = next(generator)
            #while 1:
            for train_batch in train_batch_generator:
                seq_source_ids = train_batch['seq_source_ids']
                seq_source_words = train_batch['seq_source_words']
                seq_source_len = train_batch['seq_source_len']
                seq_ref_ids = train_batch['seq_ref_ids']
                seq_ref_words = train_batch['seq_ref_words']
                seq_ref_len = train_batch['seq_ref_len']

                #debug_data(seq_source_ids, seq_ref_ids, seq_source_len, seq_ref_len, id_to_vocab)
                #return

                feed_dict = {
                    model['lr']: lr,
                    model['seq_source_ids']: seq_source_ids,
                    model['seq_source_lengths']: seq_source_len,
                    model['seq_reference_ids']: seq_ref_ids,
                    model['seq_reference_lengths']: seq_ref_len
                }

                feeds = [
                    model['train_step'], 
                    model['loss'], 
                    model['predictions'], 
                    model['logits'], 
                    model['summaries'],
                    model['final_sequence_lengths']
                ]

                try:
                    _, batch_loss, predictions, logits, summary, fsl = sess.run(feeds, feed_dict)
                except Exception as e:
                    debug_data(seq_source_ids, seq_ref_ids, seq_source_len, seq_ref_len, id_to_vocab)
                    raise e

                train_losses.append(batch_loss)

                if global_step % 25 == 0:
                    train_writer.add_summary(summary, global_step)
                    train_writer.flush()
                    seq_ref_words = [ [ref_words] for ref_words in seq_ref_words ]
                    bleu_pred_words = [ [ id_to_vocab[vocab_id] for vocab_id in prediction if vocab_id in id_to_vocab ] for prediction in predictions ]
                    bleu_pred_words = [ pred_words[:pred_words.index('<END>') if '<END>' in pred_words else len(pred_words) ] for pred_words in bleu_pred_words ]
                    bleu_score = corpus_bleu(seq_ref_words, bleu_pred_words, smoothing_function=chencherry.method1)
                    summarize_scalar(train_writer, 'bleu_score', bleu_score, global_step)
                    train_loss = sum(train_losses) / len(train_losses)
                    summarize_scalar(train_writer, 'loss', train_loss, global_step)
                    print("{} : step={} epoch={} batch_loss={:.4f} train_loss={:.4f} bleu={:.4f}".format(dt.datetime.now(), global_step, epoch, batch_loss, train_loss, bleu_score), flush=True)

                if global_step % 1000 == 0:
                    debug_data(seq_source_ids, seq_ref_ids, seq_source_len, seq_ref_len, id_to_vocab)
                    print("PREDICTIONS!")
                    print("logits shape: " + str(logits.shape))
                    print("final_seq_lengths: " + str(fsl))
                    print("len(predictions): " + str(len(predictions)))
                    #print("predictions: " + str(predictions))
                    for prediction in predictions:
                        print(str(len(prediction)) + ' ' + ' '.join([id_to_vocab[vocab_id] for vocab_id in prediction if vocab_id in id_to_vocab]))

                if global_step % 1000 == 0 and global_step != 0:
                    #dev_batch_generator = dataset_generator.generate_batch(args.batch_size, 'dev')
                    dev_batch_generator = dataset_generator.generate_batch(64, 'dev')
                    dev_batch_losses = []
                    dev_predictions = []
                    dev_seq_ref_words = []
                    dev_bleu_pred_words = []
                    for dev_batch in dev_batch_generator:
                        seq_source_ids = dev_batch['seq_source_ids']
                        seq_source_words = dev_batch['seq_source_words']
                        seq_source_len = dev_batch['seq_source_len']
                        seq_ref_ids = dev_batch['seq_ref_ids']
                        seq_ref_words = dev_batch['seq_ref_words']
                        seq_ref_len = dev_batch['seq_ref_len']

                        feed_dict = {
                            model['seq_source_ids']: seq_source_ids,
                            model['seq_source_lengths']: seq_source_len,
                            model['seq_reference_ids']: seq_ref_ids,
                            model['seq_reference_lengths']: seq_ref_len
                        }

                        feeds = [
                            model['loss'],
                            model['predictions'], 
                            model['summaries'],
                            model['final_sequence_lengths']
                        ]
                        
                        batch_loss, predictions, summary, fsl = sess.run(feeds, feed_dict)
                        dev_batch_losses.append(batch_loss)
                        dev_predictions.extend(predictions)
                        seq_ref_words = [ [ref_words] for ref_words in seq_ref_words ]
                        dev_seq_ref_words.extend(seq_ref_words)

                        bleu_pred_words = [ [ id_to_vocab[vocab_id] for vocab_id in prediction if vocab_id in id_to_vocab ] for prediction in predictions ]
                        bleu_pred_words = [ pred_words[:pred_words.index('<END>') if '<END>' in pred_words else len(pred_words) ] for pred_words in bleu_pred_words ]
                        dev_bleu_pred_words.extend(bleu_pred_words)

                        bleu_score = corpus_bleu(dev_seq_ref_words, dev_bleu_pred_words, smoothing_function=chencherry.method1)

                    summarize_scalar(dev_writer, 'bleu_score', bleu_score, global_step)
                    summarize_scalar(dev_writer, 'loss', sum(dev_batch_losses) / len(dev_batch_losses), global_step)
                    dev_writer.flush()
                    print("{} : Validation set validation_loss={:.4f} bleu={:.4f}".format(dt.datetime.now(), sum(dev_batch_losses) / len(dev_batch_losses), bleu_score), flush=True)
                
                if global_step % 5000 == 0 and global_step != 0:
                    saver.save(sess, os.path.join(train_logdir, 'model'), global_step=global_step)

                global_step += 1
                # End train batch

            saver.save(sess, os.path.join(train_logdir, 'model'), global_step=global_step)
            # End epoch

        saver.save(sess, os.path.join(train_logdir, 'model'), global_step=global_step)
        lr /= 10.
        # End sess

if __name__ == '__main__':
    main()

