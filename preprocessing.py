import 


'''
class: TokenRegistrar
purpose: call functions to create POS vector, token vector and apply model to create 
    embedding vector. Generate code to house token - embedding - sentence database for lookups
params: the BERT model
'''

'''
param: sentence - a plaintext sentence
purpose: run a set of tokenizers on a sentence
output: tokenVector - a vector of tokenized words in a sentence
'''

'''
param: sentence - a tokenized plaintext sentence
purpose: part-of-speech(POS) tag input sentence
output: posVector - a vector of POS tags equal in length to the tokenized sentence
'''

'''
param: tableName - table to write data to
param: embedding - embedding to insert into table
purpose: write the code that inserts the embedding into an sql table
output: success / failure
'''

'''
param: embedding - embedding to insert into table
param: sentence - sentence to insert into table
purpose: for each embedding, generate code to insert embedding into table with auto 
    increment id. For each sentence, generate code to insert embedding into table with 
    auto increment id. For each token, insert embedding into table with the ids 
    shared by sentence & embedding
'''

