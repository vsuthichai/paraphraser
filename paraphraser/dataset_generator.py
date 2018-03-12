import numpy as np
from keras.preprocessing.sequence import pad_sequences
from embeddings import load_sentence_embeddings
from six.moves import xrange
from six import iteritems
from random import shuffle

class ParaphraseDataset(object):
    """This class is responsible for batching the paraphrase dataset into mini batches
    for train, dev, and test.  The dataset itself must be partition into files
    beforehand and must follow this format:
    
    "source sentence\tsource sentence token ids\treference sentence\treference sentence token ids"
    
    The intraseparator is a space.  
    """

    def __init__(self, dataset_metadata, batch_size, embeddings, word_to_id, start_id, end_id, unk_id, mask_id):
        """ Constructor initialization.

        Args:
            dataset_metadata: metadata list that follows the format [
                    {
                        'maxlen': X
                        'train': training filename with sentences of length X,
                        'dev': dev filename with sentences of length X,
                        'test': test filename with sentences of length X,
                    },
                ].  Each element is a list that describes the train, dev, and
                test files for sentences of maximum length X.
            batch_size: mini batch size
            embeddings: pretrained embeddings
            word_to_id: vocabulary index
            start_id: start of sentence token id
            end_id: end of sentence token id
            unk_id: unknown token id
            mask_id: pad token id applied after the end of sentence.
        """
                
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

    def load_dataset_into_memory(self, dataset_type):
        """Load dataset into memory and partition by train, dev, and test."""

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
        """Return a generator that yields a mini batch of size self.batch_size.
        
        Args:
            dataset_type: 'train', 'test', or 'dev'
        """

        if dataset_type not in set(['train', 'test', 'dev']):
            raise ValueError("Invalid dataset type.")

        if dataset_type not in self.dataset:
            self.load_dataset_into_memory(dataset_type)

        dataset_size = len(self.dataset[dataset_type]['all_source_ids'])

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
        """ Pad a mini batch with mask_id.  This is intended to fill in any
        remaining time steps after the end of sentence tokens.

        Args:
            batch_ids: The mini batch of token ids of shape (batch_size, time_steps)
            max_len: The maximum number of time steps.

        Returns:
            a batch of samples padded with mask_id
        """
        padded_batch = np.array(pad_sequences(batch_ids, maxlen=max_len, padding='post', value=self.mask_id))
        return padded_batch


if __name__ == '__main__':
    from pprint import pprint as pp
    from utils import dataset_config
    dataset = dataset_config()
    word_to_id, idx_to_word, embeddings, start_id, end_id, unk_id, mask_id = load_sentence_embeddings()
    pd = ParaphraseDataset(dataset, 10, embeddings, word_to_id, start_id, end_id, unk_id, mask_id)
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

