'''
class: BERTModelWrapper
purpose: wraps the calling of the BERT model and the processing of inputs and outputs
    to the model for app use
'''

'''
param: tokenVector - a vector of tokenized words in a sentence
purpose: use BERT's wordpiece tokenizer to tokenize the sentence. Retain indice mappings
between whole tokens and wordpiece tokens
output: bertTokenizedVector - an input embedding vector in the BERT input format
output: wpMappingVector - vector of indices to allow the reassembly of wordpiece tokens and 
    embeddings
'''

'''
param: bertTokenizedVector - an input embedding vector in the BERT input format
purpose: creates sentence id vector out of the bertTokenizedVector
output: sentenceVector - a vector of ids indicating which sentence an indice belongs to
'''

'''
param: bertTokenizedVector - an input embedding vector in the BERT input format
param: sentenceVector - a vector of ids indicating which sentence an indice belongs to
purpose: pass the inputs through the BERT encoder and obtain final hidden layer embeddings
output: bertSentEmbedding - BERT embeddings for a sentence 
'''

'''
param: bertSentEmbedding - BERT embeddings for a sentence
param: wpMappingVector - vector of indices to allow the reassembly of wordpiece tokens and 
    embeddings
purpose: reassemble word piece into sentences. reassemble embeddings by combining embeddings
    for word pieces that belong to the same word
output: finalEmbedding - resassembled BERT embedding
'''