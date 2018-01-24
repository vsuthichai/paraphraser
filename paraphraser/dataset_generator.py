import numpy as np
from keras.preprocessing.sequence import pad_sequences
from load_sent_embeddings import load_sentence_embeddings
from six.moves import xrange
from six import iteritems
from random import shuffle

class ParaphraseDataset(object):
    def __init__(self, dataset_metadata, embeddings, word_to_id, start_id, end_id, unk_id, mask_id):
        i = 0

        # dataset
        self.lengths = sorted([ v for d in dataset_metadata for k, v in iteritems(d) if k == 'maxlen' ])
        self.dataset = {}
        for dm in dataset_metadata:
            for k, v in iteritems(dm):
                if k == 'maxlen':
                    self.dataset[v] = dm
        
        # Word embeddings, vocab
        self.embeddings = embeddings
        self.word_to_id = word_to_id
        self.vocab_size, self.embedding_size = embeddings.shape

        # Special tokens
        self.start_id = start_id
        self.end_id = end_id
        self.unk_id = unk_id
        self.mask_id = mask_id

    def pad_batch(self, batch_ids, max_len):
        padded_batch = np.array(pad_sequences(batch_ids, maxlen=max_len, padding='post', value=self.mask_id))
        return padded_batch

    def generate_batch(self, batch_size, dataset_type, length=None):
        if dataset_type not in set(['train', 'test', 'dev']):
            raise ValueError("Invalid dataset type.")

        if length == None:
            lengths = self.lengths
        else:
            lengths = [ length ]

        for length in lengths:
            with open(self.dataset[length][dataset_type], 'r') as f:
                batch_source_words = []
                batch_source_ids = []
                batch_source_len = []
                batch_ref_words = []
                batch_ref_ids = []
                batch_ref_len = []

                for line in f:
                    source_words, source_ids, ref_words, ref_ids = line.split('\t')
                    batch_source_words.append(source_words.split(' '))
                    batch_source_ids.append(source_ids.split(' '))
                    batch_ref_words.append(ref_words.split(' '))
                    batch_ref_ids.append(ref_ids.split(' '))

                    if len(batch_source_words) != batch_size:
                        continue

                    batch_source_len = [ len(source_ids) for source_ids in batch_source_ids ]
                    batch_ref_len = [ len(ref_ids) for ref_ids in batch_ref_ids ] 

                    yield {
                        'seq_source_ids': self.pad_batch(batch_source_ids, length),
                        'seq_source_words': batch_source_words,
                        'seq_source_len': batch_source_len,
                        'seq_ref_ids': self.pad_batch(batch_ref_ids, length),
                        'seq_ref_words': batch_ref_words,
                        'seq_ref_len': batch_ref_len,
                    }

                    batch_source_words = []
                    batch_source_ids = []
                    batch_source_len = []
                    batch_ref_words = []
                    batch_ref_ids = []
                    batch_ref_len = []

                if len(batch_source_words) > 0:
                    batch_source_len = [ len(source_ids) for source_ids in batch_source_ids ]
                    batch_ref_len = [ len(ref_ids) for ref_ids in batch_ref_ids ] 

                    yield {
                        'seq_source_ids': batch_source_ids,
                        'seq_source_words': batch_source_words,
                        'seq_source_len': batch_source_len,
                        'seq_ref_ids': batch_ref_ids,
                        'seq_ref_words': batch_ref_words,
                        'seq_ref_len': batch_ref_len
                    }

    def check_seq_len(self, max_tokens, batch_source_words, batch_ref_words, batch_source_ids, batch_ref_ids):
        for source_words in batch_source_words:
            del source_words[max_tokens:]
        for ref_words in batch_ref_words:
            del ref_words[max_tokens:]
        for source_ids in batch_source_ids:
            del source_ids[max_tokens:]
        for ref_ids in batch_ref_ids:
             del ref_ids[max_tokens:]

        return batch_source_words, batch_ref_words, batch_source_ids, batch_ref_ids

if __name__ == '__main__':
    from pprint import pprint as pp
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
    word_to_id, idx_to_word, embeddings, start_id, end_id, unk_id = load_sentence_embeddings()
    pd = ParaphraseDataset(dataset, embeddings, word_to_id, start_id, end_id, unk_id, mask_id=5800)
    generator = pd.generate_batch(5, 'train', 30)
    d = next(generator)

    print("=== seq source ids ===")
    print(d['seq_source_ids'].shape, flush=True)
    print(d['seq_source_ids'], flush=True)
    for i in d['seq_source_words']:
        print(i)
    print(d['seq_source_len'], flush=True)

    print("=== seq ref ids ===")
    print(d['seq_ref_ids'].shape, flush=True)
    print(d['seq_ref_ids'], flush=True)
    for i in d['seq_ref_words']:
        print(i)
    print(d['seq_ref_len'])

