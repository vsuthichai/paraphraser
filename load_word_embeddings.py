import pickle
import numpy as np

def add_start_end_tokens(vocab_to_id, id_to_vocab, embeddings):
    vocab_size, dim_size = embeddings.shape

    vocab_to_id['<unk>'] = vocab_size
    id_to_vocab[vocab_size] = '<unk>'

    vocab_to_id['<start>'] = vocab_size + 1
    id_to_vocab[vocab_size + 1] = '<start>'

    vocab_to_id['<end>'] = vocab_size + 2
    id_to_vocab[vocab_size + 2] = '<end>'

    embeddings = np.vstack((embeddings, np.random.randn(3, dim_size)))

    return vocab_to_id, id_to_vocab, embeddings

def load_glove_pickle(filename):
    if not hasattr(load_glove_pickle, 'd'):
        d = {}
        load_glove_pickle.d = d
    else:
        d = load_glove_pickle.d

    if filename in d:
        vocab_to_id = d[filename]['vocab_to_id']
        id_to_vocab = d[filename]['id_to_vocab']
        embeddings = d[filename]['embeddings']
    else:
        with open(filename, 'rb') as f:
            vocab_to_id, id_to_vocab, embeddings = pickle.load(f)
            vocab_to_id, id_to_vocab, embeddings = add_start_end_tokens(vocab_to_id, id_to_vocab, embeddings)

            d[filename] = {
                'vocab_to_id': vocab_to_id,
                'id_to_vocab': id_to_vocab,
                'embeddings': embeddings
            }


    return vocab_to_id, id_to_vocab, embeddings

def load_keras_embedding(filename):
    vocab_to_id, id_to_vocab, embeddings = load_glove_pickle(filename)
    vocab_size, hidden_size = embeddings.shape
    return Embedding(vocab_size, hidden_size, weights=embeddings, trainable=True)

