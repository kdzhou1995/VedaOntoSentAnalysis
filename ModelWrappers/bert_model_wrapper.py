from Helpers import SentenceHelpers

import torch
from transformers import BertTokenizer, BertModel

class BERTModelWrapper:

    '''
    class: BERTModelWrapper
    purpose: wraps the calling of the BERT model and the processing of inputs and outputs
        to the model for app use
    '''
    def __init__(self, modelName, outputHiddenStates = True):
        self.tokenizer = BertTokenizer.from_pretrained('modelName')
        self.model = BertModel.from_pretrained(modelName, output_hidden_states = outputHiddenStates)
        return

    '''
    param: sentence - a plain text sentence
    purpose: use BERT's wordpiece tokenizer to tokenize the sentence. Retain indice mappings
    between whole tokens and wordpiece tokens
    output: tokensTensor - an input vector of token ids in the BERT input format
    output: tokenizedSentence - sentence split by white space into words
    '''
    def TokenizeSentence(self, sentence):
        markupSentence = "[CLS]" + sentence + "[SEP]"
        tokenizedSentence = self.tokenizer.tokenize(markupSentence)
        token_ids = self.tokenizer.convert_tokens_to_ids(tokenizedSentence)
        return torch.tensor(token_ids), tokenizedSentence
    
    '''
    param: bertInputTokens- an input embedding vector in the BERT input format
    purpose: creates sentence id vector out of the bertTokenizedVector
    output: sentenceTensor - a vector of ids indicating which sentence an indice belongs to
    '''
    def GetSentenceIdVector(self, bertInputTokens):
        sentence_ids = [1] * len(bertInputTokens)
        return torch.tensor(sentence_ids)
    
    '''
    param: tokensTensor - an input embedding tensor in the BERT input format
    param: sentenceTensor - a tensor of ids indicating which sentence an indice belongs to
    purpose: pass the inputs through the BERT encoder and obtain final hidden layer embeddings
    output: bertEmbedding - BERT embeddings for a sentence 
    '''
    def GetSentenceEmbedding(self, tokensTensor, sentenceTensor):
        with torch.no_grad():
            encoderOutput = self.model(tokensTensor, sentenceTensor)

            finalHiddenState = encoderOutput[2][-1]
            
        return finalHiddenState
    
    '''
    param: bertEmbeddingOutput - BERT embeddings for a sentence
    param: wpMappingDict - dictionary of vectors of indices to allow the reassembly of wordpiece tokens and 
        embeddings
    param: averagingFunction - the function used to combine embeddings
    purpose: reassemble word piece into sentences. reassemble embeddings by combining embeddings
        for word pieces that belong to the same word
    output: bertEmbedding - resassembled BERT embedding
    '''
    def ReassembleEmbedding(self, bertEmbeddingOutput, tokenizedSentence, averagingFunction):

        if len(bertEmbeddingOutput) != len(tokenizedSentence):
            raise Exception("BERT embeddings output vector does not match tokenized sentence in length")
        
        groupedWpEmbeddings = []
        for token, embedding in zip(tokenizedSentence, bertEmbeddingOutput):
            if '##' in token:
                groupedWpEmbeddings[-1].append(embedding)
            else:
                groupedWpEmbeddings.append([embedding])

        bertEmbeddings = []
        for embeddingSet in groupedWpEmbeddings:
            bertEmbeddings.append(averagingFunction(embeddingSet))

        return torch.tensor(bertEmbeddings)
    
'''
param: groupedEmbeddingList - a list of embeddings of the same size to be averaged
purposes: combine a list of embeddings via mean/average
output: combined embeddings
'''    
def meanCombineEmbedding(groupedEmbeddingList):
    return torch.sum(groupedEmbeddingList)/len(groupedEmbeddingList)