import tensorflow as tf
from load_sent_embeddings import load_sentence_embeddings
from preprocess_data import preprocess_batch

def translate(predictions, decoder, id_to_vocab, end_id):
    if decoder == 'beam':
        _, sentence_length, num_samples = predictions.shape
        for i in xrange(num_samples):
            sent_pred = []
            for j in xrange(sentence_length):
                sent_pred.append(predictions[0][j][i])
            try:
                end_index = sent_pred.index(end_id)
                sent_pred = sent_pred[:end_index]
            except Exception as e:
                pass
            print("Paraphrase : {}".format(' '.join([ id_to_vocab[pred] for pred in sent_pred ])))
    else:
        for sent_pred in predictions:
            if sent_pred[-1] == end_id:
                sent_pred = sent_pred[0:-1]
            print("Paraphrase : {}".format(' '.join([ id_to_vocab[pred] for pred in sent_pred ])))


def infer(sess, model, decoder, id_to_vocab, end_id):
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
            model['predictions']
            #model['final_sequence_lengths']
        ]

        # predictions, final_sequence_lengths = sess.run(feeds, feed_dict)
        predictions = sess.run(feeds, feed_dict)[0][0]
        #print(predictions)
        #print(predictions.shape)
        translate(predictions, decoder, id_to_vocab, end_id)


def main():
    word_to_id, idx_to_word, embedding, start_id, end_id, unk_id  = load_sentence_embeddings()

    with open('frozen_model.pb', 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        predictions = tf.import_graph_def(
            graph_def=graph_def,
            return_elements=['predictions:0'])

        seq_source_ids = graph.get_tensor_by_name('import/placeholders/source_ids:0')
        seq_source_lengths = graph.get_tensor_by_name('import/placeholders/sequence_source_lengths:0')

        with tf.Session(graph=graph) as sess:
            model = {
                'seq_source_ids': seq_source_ids,
                'seq_source_lengths': seq_source_lengths,
                'predictions': predictions
            }

            infer(sess, model, 'sample', idx_to_word, end_id)

if __name__ == '__main__':
    main()
