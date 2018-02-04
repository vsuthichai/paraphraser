import tensorflow as tf
from load_sent_embeddings import load_sentence_embeddings
from preprocess_data import preprocess_batch
from six.moves import input

word_to_id, idx_to_word, embedding, start_id, end_id, unk_id  = load_sentence_embeddings()
mask_id = 5800

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
    'sampling_temperature': sampling_temperature
}

sess = tf.Session()

def restore_model(checkpoint):
    model = lstm_model(sess, 'infer', 300, embedding, start_id, end_id, mask_id)
    saver = tf.train.Saver()
    saver.restore(sess, checkpoint)

def translate(predictions, decoder, id_to_vocab, end_id):
    """ Translate the vocabulary ids in `predictions` to actual words
    that compose the paraphrase.

    Args:
        predictions : arrays of vocabulary ids
        decoder : 0 for greedy, 1 for sample, 2 for beam
        id_to_vocab : dict of vocabulary index to word
        end_id : end token index 

    Returns:
        str : the paraphrase
    """
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
            return ' '.join([ id_to_vocab[pred] for pred in sent_pred ])
    else:
        for sent_pred in predictions:
            if sent_pred[-1] == end_id:
                sent_pred = sent_pred[0:-1]
            return ' '.join([ id_to_vocab[pred] for pred in sent_pred ])


def infer(sess, model, decoder, source_sent, id_to_vocab, end_id, temp):
    """ Perform inferencing.  In other words, generate a paraphrase
    for the source sentence.

    Args:
        sess : Tensorflow session.
        model : dict of tensor to value
        decoder : 0 for greedy, 1 for sampling
        source_sent : source sentence to generate a paraphrase for
        id_to_vocab : dict of vocabulary index to word
        end_id : the end token
        temp : the sampling temperature to use when `decoder` is 1

    Returns:
        str : for the generated paraphrase
    """

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

    predictions = sess.run(feeds, feed_dict)[0][0]
    return translate(predictions, decoder, id_to_vocab, end_id)

def greedy_paraphrase(sentence):
    """Paraphrase using greedy sampler
    
    Args:
        sentence : The source sentence to be paraphrased.

    Returns:
        str : a candidate paraphrase of the `sentence`
    """

    with tf.Session(graph=graph) as sess:
        return infer(sess, model, 0, sentence, idx_to_word, end_id, 0.)

def sampler_paraphrase(sentence, sampling_temp=1.0):
    """Paraphrase by sampling a distribution

    Args:
        sentence (str): A sentence input that will be paraphrased by 
            sampling from distribution.
        sampling_temp (int) : A number between 0 an 1

    Returns:
        str: a candidate paraphrase of the `sentence`
    """

    with tf.Session(graph=graph) as sess:
        return infer(sess, model, 1, sentence, idx_to_word, end_id, sampling_temp)

def main():
    while 1:
        source_sentence = input("Source: ")
        #print("Paraph: {}".format(sampler_paraphrase('hello world.')))
        print("Paraph: {}".format(greedy_paraphrase('hello world.')))

if __name__ == '__main__':
    main()


