import numpy as np
from keras.preprocessing.sequence import pad_sequences
from load_sent_embeddings import load_sentence_embeddings
from six.moves import xrange
from six import iteritems
from random import shuffle

class ParaphraseDataset(object):
    def __init__(self, dataset_metadata, batch_size, embeddings, word_to_id, start_id, end_id, unk_id, mask_id):
        # batch size
        self.batch_size = batch_size

        # Special tokens
        self.start_id = start_id
        self.end_id = end_id
        self.unk_id = unk_id
        self.mask_id = mask_id
        
        # Word embeddings, vocab
        self.embeddings = embeddings
        self.word_to_id = word_to_id
        self.vocab_size, self.embedding_size = embeddings.shape

        # dataset
        self.lengths = sorted([ v for d in dataset_metadata for k, v in iteritems(d) if k == 'maxlen' ])
        self.dataset_metadata = {}
        for dm in dataset_metadata:
            for k, v in iteritems(dm):
                if k == 'maxlen':
                    self.dataset_metadata[v] = dm
        self.dataset = {}
        #self.load_dataset_into_memory('train')
        #self.load_dataset_into_memory('dev')
        #self.load_dataset_into_memory('test')

    def load_dataset_into_memory(self, dataset_type):
        if dataset_type not in set(['train', 'test', 'dev']):
            raise ValueError("Invalid dataset type.")

        self.dataset[dataset_type] = {}
        self.dataset[dataset_type]['all_source_words'] = []
        self.dataset[dataset_type]['all_source_ids'] = []
        self.dataset[dataset_type]['all_source_len'] = []
        self.dataset[dataset_type]['all_ref_words'] = []
        self.dataset[dataset_type]['all_ref_ids'] = []
        self.dataset[dataset_type]['all_ref_len'] = []

        batch_source_words = []
        batch_source_ids = []
        batch_source_len = []
        batch_ref_words = []
        batch_ref_ids = []
        batch_ref_len = []

        for length in self.lengths:
            with open(self.dataset_metadata[length][dataset_type], 'r') as f:
                for i, line in enumerate(f):
                    source_words, source_ids, ref_words, ref_ids = line.split('\t')
                    batch_source_words.append(source_words.strip().split(' '))
                    batch_source_ids.append(source_ids.strip().split(' '))
                    batch_ref_words.append(ref_words.strip().split(' '))
                    batch_ref_ids.append(ref_ids.strip().split(' '))

                    if i % self.batch_size != 0 and i != 0:
                        continue

                    batch_source_len = [ len(source_ids) for source_ids in batch_source_ids ]
                    batch_ref_len = [ len(ref_ids) for ref_ids in batch_ref_ids ] 

                    self.dataset[dataset_type]['all_source_ids'].append(self.pad_batch(batch_source_ids, length))
                    self.dataset[dataset_type]['all_source_words'].append(batch_source_words)
                    self.dataset[dataset_type]['all_source_len'].append(batch_source_len)
                    self.dataset[dataset_type]['all_ref_ids'].append(self.pad_batch(batch_ref_ids, length))
                    self.dataset[dataset_type]['all_ref_words'].append(batch_ref_words)
                    self.dataset[dataset_type]['all_ref_len'].append(batch_ref_len)

                    batch_source_words = []
                    batch_source_ids = []
                    batch_source_len = []
                    batch_ref_words = []
                    batch_ref_ids = []
                    batch_ref_len = []

                if len(batch_source_words) > 0:
                    batch_source_len = [ len(source_ids) for source_ids in batch_source_ids ]
                    batch_ref_len = [ len(ref_ids) for ref_ids in batch_ref_ids ] 

                    self.dataset[dataset_type]['all_source_ids'].append(self.pad_batch(batch_source_ids, length))
                    self.dataset[dataset_type]['all_source_words'].append(batch_source_words)
                    self.dataset[dataset_type]['all_source_len'].append(batch_source_len)
                    self.dataset[dataset_type]['all_ref_ids'].append(self.pad_batch(batch_ref_ids, length))
                    self.dataset[dataset_type]['all_ref_words'].append(batch_ref_words)
                    self.dataset[dataset_type]['all_ref_len'].append(batch_ref_len)

                    batch_source_words = []
                    batch_source_ids = []
                    batch_source_len = []
                    batch_ref_words = []
                    batch_ref_ids = []
                    batch_ref_len = []

    def generate_batch(self, dataset_type):
        if dataset_type not in set(['train', 'test', 'dev']):
            raise ValueError("Invalid dataset type.")

        if dataset_type not in self.dataset:
            self.load_dataset_into_memory(dataset_type)

        dataset_size = len(self.dataset[dataset_type]['all_source_ids'])
        print(dataset_size)

        rs = np.random.get_state()
        np.random.shuffle(self.dataset[dataset_type]['all_source_ids'])
        np.random.set_state(rs)
        np.random.shuffle(self.dataset[dataset_type]['all_source_words'])
        np.random.set_state(rs)
        np.random.shuffle(self.dataset[dataset_type]['all_source_len'])
        np.random.set_state(rs)
        np.random.shuffle(self.dataset[dataset_type]['all_ref_ids'])
        np.random.set_state(rs)
        np.random.shuffle(self.dataset[dataset_type]['all_ref_words'])
        np.random.set_state(rs)
        np.random.shuffle(self.dataset[dataset_type]['all_ref_len'])
        np.random.set_state(rs)

        for i in xrange(dataset_size):
            yield {
                'seq_source_ids': self.dataset[dataset_type]['all_source_ids'][i],
                'seq_source_words': self.dataset[dataset_type]['all_source_words'][i],
                'seq_source_len': self.dataset[dataset_type]['all_source_len'][i],
                'seq_ref_ids': self.dataset[dataset_type]['all_ref_ids'][i],
                'seq_ref_words': self.dataset[dataset_type]['all_ref_words'][i],
                'seq_ref_len': self.dataset[dataset_type]['all_ref_len'][i]
            }

    def pad_batch(self, batch_ids, max_len):
        padded_batch = np.array(pad_sequences(batch_ids, maxlen=max_len, padding='post', value=self.mask_id))
        return padded_batch

    '''
    def generate_batch(self, dataset_type, length=None):
        if dataset_type not in set(['train', 'test', 'dev']):
            raise ValueError("Invalid dataset type.")

        if length == None:
            lengths = self.lengths
        else:
            lengths = [ length ]

        for length in lengths:
            with open(self.dataset_metadata[length][dataset_type], 'r') as f:
                batch_source_words = []
                batch_source_ids = []
                batch_source_len = []
                batch_ref_words = []
                batch_ref_ids = []
                batch_ref_len = []

                for line in f:
                    source_words, source_ids, ref_words, ref_ids = line.split('\t')
                    batch_source_words.append(source_words.strip().split(' '))
                    batch_source_ids.append(source_ids.strip().split(' '))
                    batch_ref_words.append(ref_words.strip().split(' '))
                    batch_ref_ids.append(ref_ids.strip().split(' '))

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
                        'seq_source_ids': self.pad_batch(batch_source_ids, length),
                        'seq_source_words': batch_source_words,
                        'seq_source_len': batch_source_len,
                        'seq_ref_ids': self.pad_batch(batch_ref_ids, length),
                        'seq_ref_words': batch_ref_words,
                        'seq_ref_len': batch_ref_len
                    }
    '''

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
    pd = ParaphraseDataset(dataset, 10, embeddings, word_to_id, start_id, end_id, unk_id, mask_id=5800)
    generator = pd.generate_batch('train')
    for i, d in enumerate(generator):
        if i == 5:
            break
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

