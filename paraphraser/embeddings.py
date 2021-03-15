import numpy as np
import pickle
from six import iteritems
from pprint import pprint as pp
#from keras.layers.embeddings import Embedding

def load_sentence_embeddings():
    '''Load John Wieting sentence embeddings'''
    with open("../para-nmt-50m/ngram-word-concat-40.pickle", 'rb') as f:
        # [ numpy.ndarray(95283, 300), numpy.ndarray(74664, 300), (trigram_dict, word_dict)]
        x = pickle.load(f, encoding='latin1')
        word_vocab_size, embedding_size = x[1].shape

        trigram_embeddings, word_embeddings, _ = x
        trigram_to_id, word_to_id = x[2]

        word_to_id['<START>'] = word_vocab_size
        word_to_id['<END>'] = word_vocab_size + 1

        idx_to_word = { idx: word for word, idx in iteritems(word_to_id) }

        word_embeddings = np.vstack((word_embeddings, np.random.randn(2, embedding_size)))

        return (word_to_id, idx_to_word, word_embeddings, word_to_id['<START>'], 
               word_to_id['<END>'], word_to_id['UUUNKKK'], word_to_id['★'])

def load_glove_embeddings():
    with open("/media/sdb/datasets/glove.6B/glove.6B.300d.pickle", "rb") as f:
        word_to_id, id_to_word, word_embeddings = pickle.load(f, encoding='latin1')
        word_vocab_size, embedding_size = word_embeddings.shape
        word_to_id['<START>'] = word_vocab_size
        word_to_id['<END>'] = word_vocab_size + 1
        word_to_id['UUUNKKK'] = word_vocab_size + 2
        word_to_id['★'] = word_vocab_size + 3
        id_to_word[word_vocab_size] = '<START>'
        id_to_word[word_vocab_size+1] = '<END>'
        id_to_word[word_vocab_size+2] = 'UUUNKKK'
        id_to_word[word_vocab_size+3] = '★'
        word_embeddings = np.vstack((word_embeddings, np.random.randn(4, embedding_size)))
        return (word_to_id, id_to_word, word_embeddings, word_to_id['<START>'], 
               word_to_id['<END>'], word_to_id['UUUNKKK'], word_to_id['★'])
        

if __name__ == '__main__':
    word_to_id, idx_to_word, embedding, start_id, end_id, unk_id, mask_id = load_sentence_embeddings()
    pp(idx_to_word[mask_id])
    #pp(idx_to_word)
    #pp(word_to_id)
    #print(embedding.shape)

