import argparse
import tensorflow as tf
import numpy as np
import os
import sys
import datetime as dt
from six.moves import xrange, input
from lstm_model_beam import lstm_model
from embeddings import load_sentence_embeddings
from dataset_generator import ParaphraseDataset
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction
from utils import dataset_config, debug_data, summarize_scalar
import logging

logging.basicConfig(format = u'[%(asctime)s] %(levelname)-8s : %(message)s', level = logging.INFO)

def evaluate(sess, model, dataset_generator, mode, id_to_vocab):
    """Evaluate current model on the dev or test set.
    
    Args:
        sess: Tensorflow session
        model: dictionary containing model's tensors of interest for evaluation
        dataset_generator: dataset batch generator
        mode: 'dev' or 'test'
        id_to_vocab: voabulary dictionary id -> word

    Returns:
        loss: the loss after evaluating the dataset
        bleu_score: BLEU score after evaluation
    """

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
    logging.info("{} : Evaluating on {} set loss={:.4f} bleu={:.4f}".format(dt.datetime.now(), mode, loss, bleu_score))
    return loss, bleu_score

def infer(sess, args, model, id_to_vocab, end_id):
    """Perform inference on a model.  This is intended to be interactive.
    A user will run this from the command line to provide an input sentence
    and receive a paraphrase as output continuously within a loop.

    Args:
        sess: Tensorflow session
        args: ArgumentParser object configuration
        model: a dictionary containing the model tensors
        id_to_vocab: vocabulary index of id_to_vocab
        end_id: the end of sentence token

    """
    from preprocess_data import preprocess_batch

    while 1:
        source_sent = input("Enter source sentence: ")
        seq_source_words, seq_source_ids = preprocess_batch([ source_sent ])
        seq_source_len = [ len(seq_source) for seq_source in seq_source_ids ]

        if args.decoder == 'greedy':
            decoder = 0
        elif args.decoder == 'sample':
            decoder = 1

        feed_dict = {
            model['seq_source_ids']: seq_source_ids,
            model['seq_source_lengths']: seq_source_len,
            model['decoder_technique']: decoder,
            model['sampling_temperature']: args.sampling_temperature,
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
            
def compress_graph(sess, args, model):
    """After training has completed, this function can be called to compress
    the model.  The computation graph is frozen turning the checkpoint
    variables into constants.  Finally, optimization is done by stripping
    away all unnecessary nodes from the graph if they are not used at
    inference time.

    Args:
        sess: Tensorflow session
        args: ArgumentParser config object
        model: model dictionary containing tensors of interest

    """
    from tensorflow.python.tools import freeze_graph 
    from tensorflow.python.tools import optimize_for_inference_lib

    tf.train.write_graph(sess.graph_def, '/media/sdb/models/paraphraser', 'model.pb', as_text=False)

    freeze_graph.freeze_graph(
        #input_graph='/tmp/model.pbtxt', 
        input_graph='/media/sdb/models/paraphraser/model.pb',
        input_saver='',
        input_binary=True, 
        input_checkpoint=args.checkpoint,
        output_node_names='predictions',
        restore_op_name='save/restore_all', 
        filename_tensor_name='save/Const:0',
        output_graph='/media/sdb/models/paraphraser/frozen_model.pb', 
        clear_devices=True, 
        initializer_nodes='')

    '''
    input_graph_def = tf.GraphDef()
    #with tf.gfile.Open('/media/sdb/models/paraphraser/frozen_model.pb', 'rb') as f:
    with tf.gfile.Open('/tmp/frozen_model.pb', 'rb') as f:
        data = f.read()
        input_graph_def.ParseFromString(data)
        with tf.Graph().as_default() as graph:
            tf.import_graph_def(input_graph_def)
            print(dir(graph))
            print(graph.find_tensor_by_name('placeholders/sampling_temperature'))

    output_graph_def = optimize_for_inference_lib.optimize_for_inference(
        input_graph_def,
        ['placeholders/source_ids', 'placeholders/sequence_source_lengths'],
        ['predictions'],
        tf.float32.as_datatype_enum)
    
    f = tf.gfile.FastGFile('/tmp/optimized_model.pb', "w")
    f.write(output_graph_def.SerializeToString())
    '''

        
def parse_arguments():
    """Argument parser configuration."""
    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir', type=str, default="logs", help="Log directory to store tensorboard summary and model checkpoints")
    parser.add_argument('--epochs', type=int, default=3, help="Number of epochs to train")
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate")
    parser.add_argument('--batch_size', type=int, default=64, help="Mini batch size")
    parser.add_argument('--max_seq_length', type=int, default=40, help="Maximum sequence length.  Sentence lengths beyond this are truncated.")
    parser.add_argument('--hidden_size', type=int, default=300, help="Hidden dimension size")
    parser.add_argument('--keep_prob', type=float, default=0.8, help="Keep probability for dropout")
    parser.add_argument('--decoder', type=str, choices=['greedy', 'sample'], help="Decoder type")
    parser.add_argument('--sampling_temperature', type=float, default=0.0, help="Sampling temperature")
    parser.add_argument('--mode', type=str, default=None, choices=['train', 'dev', 'test', 'infer'], help='train or dev or test or infer or minimize')
    parser.add_argument('--checkpoint', type=str, default=None, help="Model checkpoint file")
    parser.add_argument('--minimize_graph', type=bool, default=False, help="Save existing checkpoint to minimal graph")

    return parser.parse_args()

def main():
    """Entry point for all training, evaluation, and model compression begins here"""
    args = parse_arguments()
    word_to_id, id_to_vocab, embeddings, start_id, end_id, unk_id, mask_id = load_sentence_embeddings()
    vocab_size, embedding_size = embeddings.shape
    lr = args.lr

    dataset = dataset_config()

    if args.mode not in set(['train', 'dev', 'test', 'infer', 'minimize']):
        raise ValueError("{} is not a valid mode".format(args.mode))

    with tf.Session() as sess:
        start = dt.datetime.now()
        model = lstm_model(sess, args.mode, args.hidden_size, embeddings, start_id, end_id, mask_id)

        # Saver object
        saver = tf.train.Saver()
        name_to_var_map = {var.op.name: var for var in tf.global_variables()}

        # Restore checkpoint
        if args.checkpoint:
            saver.restore(sess, args.checkpoint)

        # Save minimal graph
        if args.minimize_graph:
            compress_graph(sess, args, model)
            return

        # Load dataset only in train, dev, or test mode
        if args.mode in set(['train', 'dev', 'test']):
            logging.info("{}: Loading dataset into memory.".format(dt.datetime.now()))
            dataset_generator = ParaphraseDataset(dataset, args.batch_size, embeddings, word_to_id, start_id, end_id, unk_id, mask_id)

        # Evaluate on dev or test
        if args.mode == 'dev' or args.mode == 'test':
            evaluate(sess, model, dataset_generator, args.mode, id_to_vocab)
            return

        # Perform inferencing
        if args.mode == 'infer':
            infer(sess, args, model, id_to_vocab, end_id)
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

        chencherry = SmoothingFunction()
        global_step = 0
        tf.global_variables_initializer().run()
        sess.run(model['dummy'], {model['sampling_temperature']: 7.5})

        # Training per epoch
        for epoch in xrange(args.epochs):
            train_losses = []
            train_batch_generator = dataset_generator.generate_batch('train')
            for train_batch in train_batch_generator:
                seq_source_ids = train_batch['seq_source_ids']
                seq_source_words = train_batch['seq_source_words']
                seq_source_len = train_batch['seq_source_len']
                seq_ref_ids = train_batch['seq_ref_ids']
                seq_ref_words = train_batch['seq_ref_words']
                seq_ref_len = train_batch['seq_ref_len']

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
                    model['summaries'],
                    model['final_sequence_lengths']
                ]

                try:
                    _, batch_loss, predictions, summary, fsl = sess.run(feeds, feed_dict)
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
                    logging.info("step={} epoch={} batch_loss={:.4f} train_loss={:.4f} bleu={:.4f}".format(global_step, epoch, batch_loss, train_loss, bleu_score))

                # Print predictions for this batch every 1000 steps
                # Evaluate on dev set
                if global_step % 1000 == 0 and global_step != 0:
                    debug_data(seq_source_ids, seq_ref_ids, seq_source_len, seq_ref_len, id_to_vocab)
                    logging.info("PREDICTIONS!")
                    logging.info("final_seq_lengths: " + str(fsl))
                    logging.info("len(predictions): " + str(len(predictions)))
                    for prediction in predictions:
                        logging.info(str(len(prediction)) + ' ' + ' '.join([id_to_vocab[vocab_id] for vocab_id in prediction if vocab_id in id_to_vocab]))

                    dev_loss, bleu_score = evaluate(sess, model, dataset_generator, 'dev', id_to_vocab)
                    summarize_scalar(dev_writer, 'bleu_score', bleu_score, global_step)
                    summarize_scalar(dev_writer, 'loss', dev_loss, global_step)
                    dev_writer.flush()

                # Checkpoint.
                #if global_step % 50 == 0 and global_step != 0:
                if global_step % 5000 == 0 and global_step != 0:
                    saver.save(sess, os.path.join(train_logdir, 'model'), global_step=global_step)

                global_step += 1
            # End train batch

            saver.save(sess, os.path.join(train_logdir, 'model'), global_step=global_step)
            lr /= 10.
        # End epoch

        evaluate(sess, model, dataset_generator, 'test', id_to_vocab)
    # End sess

if __name__ == '__main__':
    main()

