from config import Config
from preprocessing import *
# organize data

## - raw data
## - seed words
##      - Affect lexicon: scherer's 36 categories
##      - Appraisal: frijda's theory of emotional process
# organize lexicons and dictionaries
## - WordNet
## - BERT embeddings

# build lexicon
## - all
## - by topic category

def initialize():

    # data_preprocessing(data) : data
    #   clean data
    #   correct misspelled words
    #   tokenize
    #   discard too short
    #   too long
    #   remove punctuation numbers and stop words
    #       
    #   stem
    #   unigram? bigram?
    data = open(Config.raw_data_path, "r")
    
    count = 0
    for line in data:
        print(line)
        count += 1
        if count == 10:
            break
        
    # buildlexicon(data, lexical source, seed, expand function): lexicon

    #   seed lexicon(seed) : seeded lexicon

    #   expand lexicon(data, seeded lexicon, expand lexicon)

    return

if __name__ == "__main__":
    initialize()