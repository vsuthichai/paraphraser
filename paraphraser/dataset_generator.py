import numpy as np
from nlp_pipeline import nlp_pipeline, openmp_nlp_pipeline
from keras.preprocessing.sequence import pad_sequences
from load_sent_embeddings import load_sentence_embeddings
from six.moves import xrange
from random import shuffle

class ParaphraseDataset(object):
    def __init__(self, path, embeddings, word_to_id, start_id, end_id, unk_id, mask_id, max_encoder_tokens, max_decoder_tokens):
        i = 0
        
        # Shuffle dataset
        with open(path, 'r') as f:
            self.lines = [ line for line in f ]
        shuffle(self.lines)

        # Create validation set
        self.validation_set = self.lines[-256:]
        self.lines = self.lines[:-256]

        # Main training dataset
        self.dataset_size = len(self.lines)
        self.validation = len(self.validation_set)

        # Path to dataset
        self.path = path

        # Word embeddings, vocab
        self.embeddings = embeddings
        self.word_to_id = word_to_id
        self.vocab_size, self.embedding_size = embeddings.shape

        # Special tokens
        self.start_id = start_id
        self.end_id = end_id
        self.unk_id = unk_id
        self.mask_id = mask_id
        self.max_encoder_tokens = max_encoder_tokens
        self.max_decoder_tokens = max_decoder_tokens

    def generate_dev_batch_tf(self, batch_size):
        batch_source = []
        batch_ref = []

        for i, line in enumerate(self.validation_set):
            source, ref = line.split('\t')

            batch_source.append(source.strip())
            batch_ref.append(ref.strip())

            if len(batch_source) % batch_size != 0:
                continue

            batch_source_ids, batch_source = openmp_nlp_pipeline(batch_source, self.word_to_id, self.unk_id)
            batch_ref_ids_, batch_ref = openmp_nlp_pipeline(batch_ref, self.word_to_id, self.unk_id)

            batch_source, batch_ref, batch_source_ids, batch_ref_ids_ = self.check_seq_len(self.max_encoder_tokens, batch_source, batch_ref, batch_source_ids, batch_ref_ids_)

            batch_ref_ids = [ [self.start_id] + ref_ids + [self.end_id] for ref_ids in batch_ref_ids_ ]

            batch_source_len = [ len(source_ids) for source_ids in batch_source_ids ]
            batch_ref_len = [ len(ref_ids) for ref_ids in batch_ref_ids ] 

            encoder_input_data, decoder_input_data = self.encode_batch(batch_size, batch_source_ids, batch_ref_ids)

            yield {
                'seq_source_ids': encoder_input_data,
                'seq_ref_ids': decoder_input_data,
                'seq_ref_words': batch_ref,
                'seq_source_len': batch_source_len,
                'seq_ref_len': batch_ref_len,
            }
            #yield (encoder_input_data, decoder_input_data, one_hot_decoder_target_data, batch_source_len, batch_ref_len, batch_target_len)

            batch_source = []
            batch_ref = []

    def generate_batch_tf(self, batch_size):
        batch_source = []
        batch_ref = []

        for i, line in enumerate(self.lines):
            source, ref = line.split('\t')

            batch_source.append(source.strip())
            batch_ref.append(ref.strip())

            if len(batch_source) % batch_size != 0:
                continue

            # NLP Pipleine
            batch_source_ids, batch_source_words = openmp_nlp_pipeline(batch_source, self.word_to_id, self.unk_id)
            batch_ref_ids_, batch_ref_words = openmp_nlp_pipeline(batch_ref, self.word_to_id, self.unk_id)

            # Create reference, preprend start id, append end id
            batch_ref_ids = [ [self.start_id] + ref_ids + [self.end_id] for ref_ids in batch_ref_ids_ ]

            # Check sequence length, truncate if it exceeds max len
            batch_source_words, batch_ref_words, batch_source_ids, batch_ref_ids = self.check_seq_len(self.max_encoder_tokens, batch_source_words, batch_ref_words, batch_source_ids, batch_ref_ids)

            batch_source_len = [ len(source_ids) for source_ids in batch_source_ids ]
            batch_ref_len = [ len(ref_ids) for ref_ids in batch_ref_ids ] 

            encoder_input_data, decoder_input_data = self.encode_batch(batch_source_ids, batch_ref_ids)

            yield {
                'seq_source_ids': encoder_input_data,
                'seq_source_words': batch_source_words,
                'seq_source_len': batch_source_len,
                'seq_ref_ids': decoder_input_data,
                'seq_ref_words': batch_ref,
                'seq_ref_len': batch_ref_len,
            }
            #yield (encoder_input_data, decoder_input_data, one_hot_decoder_target_data, batch_source_len, batch_ref_len, batch_target_len)

            batch_source = []
            batch_ref = []

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

    def encode_batch(self, source_ids, ref_ids):
        encoder_input_data = np.array(pad_sequences(source_ids, maxlen=self.max_encoder_tokens, padding='post', value=self.mask_id))
        decoder_input_data = np.array(pad_sequences(ref_ids, maxlen=self.max_decoder_tokens, padding='post', value=self.mask_id))

        return encoder_input_data, decoder_input_data
        #return encoder_input_data, decoder_input_data, np.expand_dims(decoder_target_data, -1)

        #one_hot_decoder_target_data = (np.arange(self.vocab_size) == decoder_target_data[...,None]).astype(int)
        #return encoder_input_data, decoder_input_data, one_hot_decoder_target_data

if __name__ == '__main__':
    import datetime as dt
    word_to_id, idx_to_word, embeddings, start_id, end_id, unk_id = load_sentence_embeddings()
    dataset = ParaphraseDataset('/media/sdb/datasets/para-nmt-5m-processed/para-nmt-5m-processed.txt', embeddings, word_to_id)
    d = dataset.generate_batch(1, start_id, end_id, unk_id, 5800, 30, 30)
    for [encoder_input_data, decoder_input_data], one_hot_decoder_target_data in d:
        print(encoder_input_data.shape)
        print(encoder_input_data, flush=True)
        print(decoder_input_data.shape)
        print(decoder_input_data, flush=True)
        print(one_hot_decoder_target_data.shape, flush=True)
        print(one_hot_decoder_target_data, flush=True)
        break

