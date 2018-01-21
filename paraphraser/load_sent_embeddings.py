import numpy as np
import pickle
from pprint import pprint as pp
from six import iteritems
from keras.layers.embeddings import Embedding

def load_sentence_embeddings():
    with open("/home/victor/github/para-nmt-50m/data/ngram-word-concat-40.pickle", 'rb') as f:
        # [ numpy.ndarray(95283, 300), numpy.ndarray(74664, 300), (trigram_dict, word_dict)]
        x = pickle.load(f, encoding='latin1')
        word_vocab_size, embedding_size = x[1].shape

        trigram_embeddings, word_embeddings, _ = x
        trigram_to_id, word_to_id = x[2]

        word_to_id['<START>'] = word_vocab_size
        word_to_id['<END>'] = word_vocab_size + 1

        idx_to_word = { idx: word for word, idx in iteritems(word_to_id) }

        word_embeddings = np.vstack((word_embeddings, np.random.randn(2, embedding_size)))

        return word_to_id, idx_to_word, word_embeddings, word_to_id['<START>'], word_to_id['<END>'], word_to_id['UUUNKKK']

def load_keras_embedding():
    word_to_id, idx_to_word, embeddings, start_id, end_id, unk_id  = load_sentence_embeddings()
    vocab_size, embedding_size = embeddings.shape
    return Embedding(vocab_size, embedding_size, weights=[embeddings], trainable=True), vocab_size, embedding_size, start_id, end_id, unk_id

if __name__ == '__main__':
    word_to_id, idx_to_word, embedding, start_id, end_id, unk_id  = load_sentence_embeddings()
    from pprint import pprint as pp
    pp(word_to_id)
