import sys
import spacy
from spacy.tokenizer import Tokenizer
import datetime as dt
import multiprocessing as mp

nlp = spacy.load('en_core_web_sm')
tokenizer = Tokenizer(nlp.vocab)

def nlp_pipeline(sentence, word_to_id, unk_id):
    ''' Convert word tokens into their vocab ids '''
    return [ word_to_id.get(token.lower_, unk_id) for token in nlp_pipeline_0(sentence) ]

def nlp_pipeline_0(sentence):
    ''' Execute spacy pipeline, single thread '''
    return nlp(sentence, disable=['parser', 'tagger', 'ner'])

def mp_nlp_pipeline(pool, lines):
    ''' Execute spacy pipeline, multiprocessing style '''
    return pool.map(nlp_pipeline_0, lines, 1)

def openmp_nlp_pipeline(lines, n_threads=12):
    ''' Execute spacy's openmp nlp pipeline '''
    return [ [ token.lower_ for token in doc ] for doc in nlp.pipe(lines, n_threads=n_threads, disable=['parser', 'tagger', 'ner']) ]

def single_thread_nlp_pipeline(lines):
    ''' Another single thread pipeline '''
    return [ nlp(line) for line in lines ]

def main():
    import datetime as dt    
    from embeddings import load_sentence_embeddings
    #pool = mp.Pool(10)

    word_to_id, idx_to_word, embedding, start_id, end_id, unk_id = load_sentence_embeddings()
    print(unk_id)

    with open('/media/sdb/datasets/para-nmt-5m-processed/para-nmt-5m-processed.txt', 'r') as f:
        lines = []
        for i, line in enumerate(f):
            lines.append(line.strip())

            if i % 64 == 0:
                start = dt.datetime.now()
                #docs = mp_nlp_pipeline(pool, lines)
                docs = openmp_nlp_pipeline(lines, word_to_id, unk_id)
                #docs = single_thread_nlp_pipeline(lines)
                #doc = nlp_pipeline_0(line)
                print(docs)

                end = dt.datetime.now()
                print(end - start, flush=True)
                lines = []
            else:
                continue


if __name__ == '__main__':
    main()

