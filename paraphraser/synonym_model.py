import sys
import spacy
from pprint import pprint
from spacy.tokens.token import Token
from nltk.corpus import wordnet as wn
from six.moves import xrange
import random

nlp = spacy.load('en_core_web_sm')

def generate_sentence(original_doc, new_tokens):
    new_sentence = ' '.join(new_tokens).replace('_', ' ')
    new_doc = nlp(new_sentence)
    similarity_score = original_doc.similarity(new_doc)
    return (new_sentence, similarity_score)

def synonym_model(s):
    generated_sentences = set([])

    doc = nlp(s)
    original_tokens = [ token.text for token in doc ]

    index_to_lemmas = {}

    for index, token in enumerate(doc):
        index_to_lemmas[index] = set([])
        index_to_lemmas[index].add(token)

        if token.pos_ == 'NOUN' and len(token.text) >= 3:
            pos = wn.NOUN
        elif token.pos_ == 'VERB' and len(token.text) >= 3:
            pos = wn.VERB
        elif token.pos_ == 'ADV' and len(token.text) >= 3:
            pos = wn.ADV
        elif token.pos_ == 'ADJ' and len(token.text) >= 3:
            pos = wn.ADJ
        else:
            continue

        # Synsets
        for synset in wn.synsets(token.text, pos):
            for lemma in synset.lemmas():
                new_tokens = original_tokens.copy()
                new_tokens[index] = lemma.name()
                sentence_and_score = generate_sentence(doc, new_tokens)
                generated_sentences.add(sentence_and_score)
                index_to_lemmas[index].add(lemma.name())

    count = sum([ len(words) for words in index_to_lemmas.values() ])

    for i in xrange(min(count, 40)):
        new_tokens = []
        for index, words in sorted(index_to_lemmas.items(), key=lambda x: x[0]):
            token = random.sample(index_to_lemmas[index], 1)[0]
            new_tokens.append(str(token))
        sentence_and_score = generate_sentence(doc, new_tokens)
        generated_sentences.add(sentence_and_score)

    #print(generated_sentences)
    return generated_sentences

def synonym_paraphrase(s):
    return synonym_model(s)

if __name__ == '__main__':
    #x = synonym_model('I am discussing my homework with the teacher.')
    #x = synonym_model('the rabbit quickly ran down the hole')
    #x = synonym_model('John tried to fix his computer by hacking away at it.')
    x = synonym_model('team based multiplayer online first person shooter video game')
    print(x)

