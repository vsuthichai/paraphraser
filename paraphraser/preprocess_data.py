"""Dataset preprocessing and generation.

This module's purpose is to consume raw paraphrase text and output a dataset
in an optimal form to later be consumed by ParaphraseDataset class in
dataset_generator.py.  The raw text are assumed to be valid paraphrases
and must follow the following format each line:

source sentence\treference sentence

The number of tokens within a sentence are counted so that samples can be 
grouped into the same file by similar length.  After nlp preprocessing and
tokenization, the resulting new format per line is:

source sentence tokens\tsource sentence token ids\treference tokens\treference token ids

This format is consumed directly into ParaphraseDataset to generate mini
batches where each batch contains similar length sentences.

"""

import os
from six import iteritems
from nlp_pipeline import openmp_nlp_pipeline
from embeddings import load_sentence_embeddings

word_to_id, idx_to_word, embedding, start_id, end_id, unk_id, mask_id = load_sentence_embeddings()

def generate_length_index(max_lengths):
    l = []
    prev = None
    for ml in max_lengths:
        if prev == None:
            a = (ml+1) * [ml]
        else:
            a = (ml - prev) * [ml]
        prev = ml
        l.extend(a)
    return l

def word_to_token_ids(batch_docs):
    batch_token_ids = [ [ word_to_id.get(word, unk_id) for word in doc ] for doc in batch_docs ]
    return batch_token_ids

def preprocess_batch(batch_sentences):
    # NLP Pipleine
    batch_words = openmp_nlp_pipeline(batch_sentences)
    batch_ids_ = word_to_token_ids(batch_words)

    # Create reference, preprend start id, append end id
    batch_ids = [ [start_id] + ids + [end_id] for ids in batch_ids_ ]
    
    return (batch_words, batch_ids)

def fsave_data(filename, batch_source_words, batch_source_ids, batch_ref_words, batch_ref_ids):
    max_lengths = [5, 10, 20, 30, 40, 50]

    for length in max_lengths:
        try:
            os.remove(filename + "." + str(length))
        except:
            pass

    files = { length: open(filename + "." + str(length), 'a') for length in max_lengths }
    l = generate_length_index(max_lengths)

    z = zip(batch_source_words, batch_source_ids, batch_ref_words, batch_ref_ids)

    for source_words, source_ids, ref_words, ref_ids in z:
        max_len = max(len(source_ids), len(ref_ids))
        try:
            files[l[max_len]].write("{}\t{}\t{}\t{}\n".format(' '.join(source_words), 
                                                              ' '.join([ str(source_id) for source_id in source_ids ]),
                                                              ' '.join(ref_words),
                                                              ' '.join([ str(ref_id) for ref_id in ref_ids ])))
        except Exception as e:
            print(e)
            print("Error writing {} {} {} {}".format(' '.join(source_words),
                                                     ' '.join([ str(source_id) for source_id in source_ids ]),
                                                     ' '.join(ref_words),
                                                     ' '.join([ str(ref_id) for ref_id in ref_ids ])))
            continue

    for length, f in iteritems(files):
        f.close()

def preprocess_data(filename):
    batch_source_sentences = []
    batch_ref_sentences = []

    with open(filename, 'r') as f:
        for i, line in enumerate(f):
            source, ref = line.split('\t')
            batch_source_sentences.append(source.strip())
            batch_ref_sentences.append(ref.strip())

    batch_source_words, batch_source_ids = preprocess_batch(batch_source_sentences)
    batch_ref_words, batch_ref_ids = preprocess_batch(batch_ref_sentences)

    fsave_data(filename, batch_source_words, batch_source_ids, batch_ref_words, batch_ref_ids)

def main():
    import sys
    preprocess_data(sys.argv[1])

if __name__ == '__main__':
    main()

