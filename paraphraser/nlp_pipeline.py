import sys
import spacy
from spacy.tokenizer import Tokenizer

nlp = spacy.load('en')
tokenizer = Tokenizer(nlp.vocab)

def nlp_pipeline(sentence):
    tokens = nlp(sentence, disable=['parser', 'tagger', 'ner'])
    return [ token.lower_ for token in tokens ]

def main():
    print(nlp_pipeline("Most crucial moment of your lives."))

if __name__ == '__main__':
    main()

