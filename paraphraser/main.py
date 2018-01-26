import argparse
import tensorflow as tf
import numpy as np
import os
import sys
import datetime as dt
from six.moves import xrange, input
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

def evaluate(sess, model, dataset_generator, mode, id_to_vocab):
    batch_generator = dataset_generator.generate_batch(mode)
    chencherry = SmoothingFunction()
    batch_losses = []
    all_seq_ref_words = []
    all_bleu_pred_words = []

    for batch in batch_generator:
        seq_source_ids = batch['seq_source_ids']
        seq_source_words = batch['seq_source_words']
        seq_source_len = batch['seq_source_len']
        seq_ref_ids = batch['seq_ref_ids']
        seq_ref_words = batch['seq_ref_words']
        seq_ref_len = batch['seq_ref_len']

        #debug_data(seq_source_ids, seq_ref_ids, seq_source_len, seq_ref_len, id_to_vocab)
        #return

        feed_dict = {
            model['seq_source_ids']: seq_source_ids,
            model['seq_source_lengths']: seq_source_len,
            model['seq_reference_ids']: seq_ref_ids,
            model['seq_reference_lengths']: seq_ref_len
        }

        feeds = [
            model['loss'],
            model['predictions'], 
            model['final_sequence_lengths']
        ]

        try:
            batch_loss, predictions, fsl = sess.run(feeds, feed_dict)
        except Exception as e:
            debug_data(seq_source_ids, seq_ref_ids, seq_source_len, seq_ref_len, id_to_vocab)
            raise e

        #debug_data(seq_source_ids, seq_ref_ids, seq_source_len, seq_ref_len, id_to_vocab)
        #print("PREDICTIONS!")
        #print("final_seq_lengths: " + str(fsl))
        #print("len(predictions): " + str(len(predictions)))
        #print("predictions: " + str(predictions))
        #for prediction in predictions:
            #print(str(len(prediction)) + ' ' + ' '.join([id_to_vocab[vocab_id] for vocab_id in prediction if vocab_id in id_to_vocab]), flush=True)

        # batch losses
        batch_losses.append(batch_loss)

        # all ref words
        seq_ref_words = [ [ref_words] for ref_words in seq_ref_words ]
        all_seq_ref_words.extend(seq_ref_words)

        # all prediction words to compute bleu on
        bleu_pred_words = [ [ id_to_vocab[vocab_id] for vocab_id in prediction if vocab_id in id_to_vocab ] for prediction in predictions ]
        bleu_pred_words = [ pred_words[:pred_words.index('<END>') if '<END>' in pred_words else len(pred_words) ] for pred_words in bleu_pred_words ]
        all_bleu_pred_words.extend(bleu_pred_words)

    bleu_score = corpus_bleu(all_seq_ref_words, all_bleu_pred_words, smoothing_function=chencherry.method1)
    loss = sum(batch_losses) / len(batch_losses)
    print("{} : Evaluating on {} set loss={:.4f} bleu={:.4f}".format(dt.datetime.now(), mode, loss, bleu_score), flush=True)
    return loss, bleu_score

def infer(sess, model, mode, id_to_vocab, end_id):
    from preprocess_data import preprocess_batch

    while 1:
        source_sent = input("Enter source sentence: ")
        seq_source_words, seq_source_ids = preprocess_batch([ source_sent ])
        seq_source_len = [ len(seq_source) for seq_source in seq_source_ids ]

        feed_dict = {
            model['seq_source_ids']: seq_source_ids,
            model['seq_source_lengths']: seq_source_len,
        }

        feeds = [
            model['predictions'], 
            model['final_sequence_lengths']
        ]

        predictions, final_sequence_lengths = sess.run(feeds, feed_dict)

        for sent_pred in predictions:
            if sent_pred[-1] == end_id:
                sent_pred = sent_pred[0:-1]
            print("Paraphrase : {}".format(' '.join([ id_to_vocab[pred] for pred in sent_pred ])))
        

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir', type=str, default="logs", help="Log directory to store tensorboard summary and model checkpoints")
    parser.add_argument('--epochs', type=int, default=2, help="Number of epochs to train")
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate")
    parser.add_argument('--batch_size', type=int, default=64, help="Mini batch size")
    parser.add_argument('--max_seq_length', type=int, default=40, help="Maximum sequence length.  Sentence lengths beyond this are truncated.")
    parser.add_argument('--hidden_size', type=int, default=300, help="Hidden dimension size")
    parser.add_argument('--keep_prob', type=float, default=0.8, help="Keep probability for dropout")
    parser.add_argument('--beam_width', type=int, default=0, help="Beam width")
    parser.add_argument('--sampling_temperature', type=float, default=0.0, help="Sampling temperature")
    parser.add_argument('--mode', type=str, default=None, help='train or dev or test or infer')
    parser.add_argument('--checkpoint', type=str, default=None, help="Model checkpoint file")

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

    if args.mode not in set(['train', 'dev', 'test', 'infer']):
        raise ValueError("{} is not a valid mode".format(args.mode))

    with tf.Session() as sess:
        start = dt.datetime.now()
        model = lstm_model(args, embeddings, start_id, end_id, mask_id, args.mode)
        from pprint import pprint as pp
        name_to_var_map = {var.op.name: var for var in tf.global_variables()}
        name_to_var_map['decoder/decoder/attention_wrapper/bahdanau_attention/attention_v'] = name_to_var_map['decoder_1/attention_wrapper/bahdanau_attention/attention_v']
        name_to_var_map['decoder/decoder/attention_wrapper/attention_layer/kernel'] = name_to_var_map['decoder_1/attention_wrapper/attention_layer/kernel']
        name_to_var_map['decoder/decoder/attention_wrapper/bahdanau_attention/query_layer/kernel'] = name_to_var_map['decoder_1/attention_wrapper/bahdanau_attention/query_layer/kernel']
        name_to_var_map['decoder/decoder/attention_wrapper/basic_lstm_cell/bias'] = name_to_var_map['decoder_1/attention_wrapper/basic_lstm_cell/bias']
        name_to_var_map['decoder/decoder/dense/kernel'] = name_to_var_map['decoder_1/dense/kernel']
        name_to_var_map['decoder/decoder/attention_wrapper/basic_lstm_cell/kernel'] = name_to_var_map['decoder_1/attention_wrapper/basic_lstm_cell/kernel']

        del name_to_var_map['decoder_1/attention_wrapper/bahdanau_attention/attention_v']
        del name_to_var_map['decoder_1/attention_wrapper/attention_layer/kernel']
        del name_to_var_map['decoder_1/attention_wrapper/bahdanau_attention/query_layer/kernel']
        del name_to_var_map['decoder_1/attention_wrapper/basic_lstm_cell/bias']
        del name_to_var_map['decoder_1/dense/kernel']
        del name_to_var_map['decoder_1/attention_wrapper/basic_lstm_cell/kernel']

        '''
        2018-01-25 22:59:51.890576: W tensorflow/core/framework/op_kernel.cc:1192] Not found: Key decoder_1/attention_wrapper/bahdanau_attention/attention_v not found in checkpoint
        2018-01-25 22:59:51.892306: W tensorflow/core/framework/op_kernel.cc:1192] Not found: Key decoder_1/attention_wrapper/attention_layer/kernel not found in checkpoint
        2018-01-25 22:59:51.893286: W tensorflow/core/framework/op_kernel.cc:1192] Not found: Key decoder_1/attention_wrapper/bahdanau_attention/query_layer/kernel not found in checkpoint
        2018-01-25 22:59:51.893345: W tensorflow/core/framework/op_kernel.cc:1192] Not found: Key decoder_1/attention_wrapper/basic_lstm_cell/bias not found in checkpoint
        2018-01-25 22:59:51.894636: W tensorflow/core/framework/op_kernel.cc:1192] Not found: Key decoder_1/dense/kernel not found in checkpoint
        2018-01-25 22:59:51.894814: W tensorflow/core/framework/op_kernel.cc:1192] Not found: Key decoder_1/attention_wrapper/basic_lstm_cell/kernel not found in checkpoint
        '''
        '''
        decoder/decoder/attention_wrapper/attention_layer/kernel (DT_FLOAT) [1200,300]
        decoder/decoder/attention_wrapper/attention_layer/kernel/Adam (DT_FLOAT) [1200,300]
        decoder/decoder/attention_wrapper/attention_layer/kernel/Adam_1 (DT_FLOAT) [1200,300]
        decoder/decoder/attention_wrapper/bahdanau_attention/attention_v (DT_FLOAT) [300]
        decoder/decoder/attention_wrapper/bahdanau_attention/attention_v/Adam (DT_FLOAT) [300]
        decoder/decoder/attention_wrapper/bahdanau_attention/attention_v/Adam_1 (DT_FLOAT) [300]
        decoder/decoder/attention_wrapper/bahdanau_attention/query_layer/kernel (DT_FLOAT) [600,300]
        decoder/decoder/attention_wrapper/bahdanau_attention/query_layer/kernel/Adam (DT_FLOAT) [600,300]
        decoder/decoder/attention_wrapper/bahdanau_attention/query_layer/kernel/Adam_1 (DT_FLOAT) [600,300]
        decoder/decoder/attention_wrapper/basic_lstm_cell/bias (DT_FLOAT) [2400]
        decoder/decoder/attention_wrapper/basic_lstm_cell/bias/Adam (DT_FLOAT) [2400]
        decoder/decoder/attention_wrapper/basic_lstm_cell/bias/Adam_1 (DT_FLOAT) [2400]
        decoder/decoder/attention_wrapper/basic_lstm_cell/kernel (DT_FLOAT) [1200,2400]
        decoder/decoder/attention_wrapper/basic_lstm_cell/kernel/Adam (DT_FLOAT) [1200,2400]
        decoder/decoder/attention_wrapper/basic_lstm_cell/kernel/Adam_1 (DT_FLOAT) [1200,2400]
        '''

        # Saver object
        saver = tf.train.Saver(name_to_var_map)
        if args.checkpoint:
            saver.restore(sess, args.checkpoint)

        # Load dataset only in train, dev, or test mode
        if args.mode in set(['train', 'dev', 'test']):
            print("{}: Loading dataset into memory.".format(dt.datetime.now()))
            dataset_generator = ParaphraseDataset(dataset, args.batch_size, embeddings, word_to_id, start_id, end_id, unk_id, mask_id)

        # Evaluate on dev or test
        if args.mode == 'dev' or args.mode == 'test':
            evaluate(sess, model, dataset_generator, args.mode, id_to_vocab)
            return

        # Perform inferencing
        if args.mode == 'infer':
            infer(sess, model, args.mode, id_to_vocab, end_id)
            return

        ###################################
        # Training run proceeds from here #
        ###################################

        # Training summary writer
        train_logdir = os.path.join(args.log_dir, "train-" + start.strftime("%Y%m%d-%H%M%S"))
        train_writer = tf.summary.FileWriter(train_logdir)

        # Dev summary writer
        dev_logdir = os.path.join(args.log_dir, "dev-" + start.strftime("%Y%m%d-%H%M%S"))
        dev_writer = tf.summary.FileWriter(dev_logdir)

        tf.global_variables_initializer().run()
        chencherry = SmoothingFunction()
        global_step = 0

        for epoch in xrange(args.epochs):
            train_losses = []
            train_batch_generator = dataset_generator.generate_batch('train')
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

                feed_dict = {
                    model['lr']: lr,
                    model['seq_source_ids']: seq_source_ids,
                    model['seq_source_lengths']: seq_source_len,
                    model['seq_reference_ids']: seq_ref_ids,
                    model['seq_reference_lengths']: seq_ref_len,
                    model['keep_prob']: args.keep_prob
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

                # Status update
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

                # Print predictions for this batch every 1000 steps
                # Evaluate on dev set
                if global_step % 1000 == 0:
                    debug_data(seq_source_ids, seq_ref_ids, seq_source_len, seq_ref_len, id_to_vocab)
                    print("PREDICTIONS!")
                    print("logits shape: " + str(logits.shape))
                    print("final_seq_lengths: " + str(fsl))
                    print("len(predictions): " + str(len(predictions)))
                    #print("predictions: " + str(predictions))
                    for prediction in predictions:
                        print(str(len(prediction)) + ' ' + ' '.join([id_to_vocab[vocab_id] for vocab_id in prediction if vocab_id in id_to_vocab]))

                    dev_loss, bleu_score = evaluate(sess, model, dataset_generator, 'dev', id_to_vocab)
                    summarize_scalar(dev_writer, 'bleu_score', bleu_score, global_step)
                    summarize_scalar(dev_writer, 'loss', dev_loss, global_step)
                    dev_writer.flush()

                # Checkpoint.
                if global_step % 5000 == 0 and global_step != 0:
                    saver.save(sess, os.path.join(train_logdir, 'model'), global_step=global_step)

                global_step += 1
            # End train batch

            saver.save(sess, os.path.join(train_logdir, 'model'), global_step=global_step)
            lr /= 10.
        # End epoch

        evaluate(sess, model, 'test')
    # End sess

if __name__ == '__main__':
    main()

