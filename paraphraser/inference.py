import tensorflow as tf
from load_sent_embeddings import load_sentence_embeddings
from preprocess_data import preprocess_batch

word_to_id, idx_to_word, embedding, start_id, end_id, unk_id  = load_sentence_embeddings()

with open('/media/sdb/models/paraphraser/frozen_model.pb', 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

with tf.Graph().as_default() as graph:
    predictions = tf.import_graph_def(
        graph_def=graph_def,
        return_elements=['predictions:0'],
        name='')

    print([op.name for op in graph.get_operations()])

    seq_source_ids = graph.get_tensor_by_name('placeholders/source_ids:0')
    seq_source_lengths = graph.get_tensor_by_name('placeholders/sequence_source_lengths:0')
    decoder_technique = graph.get_tensor_by_name('placeholders/decoder_technique:0')
    sampling_temperature = graph.get_tensor_by_name('placeholders/sampling_temperature:0')
    keep_prob = graph.get_tensor_by_name('placeholders/keep_prob:0')

model = {
    'seq_source_ids': seq_source_ids,
    'seq_source_lengths': seq_source_lengths,
    'predictions': predictions,
    'decoder_technique': decoder_technique,
    'sampling_tempearture': sampling_temperature
}

def translate(predictions, decoder, id_to_vocab, end_id):
    if decoder == 2:
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
            #print("Paraphrase : {}".format(' '.join([ id_to_vocab[pred] for pred in sent_pred ])))
            return ' '.join([ id_to_vocab[pred] for pred in sent_pred ])
    else:
        for sent_pred in predictions:
            if sent_pred[-1] == end_id:
                sent_pred = sent_pred[0:-1]
            #print("Paraphrase : {}".format(' '.join([ id_to_vocab[pred] for pred in sent_pred ])))
            return ' '.join([ id_to_vocab[pred] for pred in sent_pred ])


def infer(sess, model, decoder, source_sent, id_to_vocab, end_id, temp):
    from preprocess_data import preprocess_batch

    seq_source_words, seq_source_ids = preprocess_batch([ source_sent ])
    seq_source_len = [ len(seq_source) for seq_source in seq_source_ids ]

    feed_dict = {
        model['seq_source_ids']: seq_source_ids,
        model['seq_source_lengths']: seq_source_len,
        model['decoder_technique']: decoder,
        model['sampling_temperature']: temp
    }

    feeds = [
        model['predictions']
        #model['final_sequence_lengths']
    ]

    # predictions, final_sequence_lengths = sess.run(feeds, feed_dict)
    predictions = sess.run(feeds, feed_dict)[0][0]
    #print(predictions)
    #print(predictions.shape)
    return translate(predictions, decoder, id_to_vocab, end_id)

def greedy_paraphrase(sentence):
    with tf.Session(graph=graph) as sess:
        return infer(sess, model, 0, sentence, idx_to_word, end_id, 0.)

def sampler_paraphrase(sentence, temp=0.5):
    with tf.Session(graph=graph) as sess:
        return infer(sess, model, 1, sentence, idx_to_word, end_id, temp)

def main():
    a = sampler_paraphrase('hello world.')
    print(a)

if __name__ == '__main__':
    main()


