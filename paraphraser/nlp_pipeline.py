import sys
import spacy
from spacy.tokenizer import Tokenizer

nlp = spacy.load('en')
tokenizer = Tokenizer(nlp.vocab)

def nlp_pipeline(sentence, word_to_id, unk_id):
    tokens = nlp(sentence, disable=['parser', 'tagger', 'ner'])
    return [ word_to_id.get(token.lower_, unk_id) for token in tokens ]

def main():
    print(nlp_pipeline("Most crucial moment of your lives.", {}, 1))

if __name__ == '__main__':
    main()

