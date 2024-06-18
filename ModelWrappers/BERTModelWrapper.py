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
        self.tokenizer = BertTokenizer.from_pretrained(modelName)
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
        return torch.tensor(token_ids)[None,:], tokenizedSentence
    
    '''
    param: bertInputTokens- an input embedding vector in the BERT input format
    purpose: creates sentence id vector out of the bertTokenizedVector
    output: sentenceTensor - a vector of ids indicating which sentence an indice belongs to
    '''
    def GetSentenceIdVector(self, bertInputTokens):
        sentence_ids = [1] * len(bertInputTokens)
        return torch.tensor(sentence_ids)[None,:]
    
    '''
    param: tokensTensor - an input embedding tensor in the BERT input format
    param: sentenceTensor - a tensor of ids indicating which sentence an indice belongs to
    purpose: pass the inputs through the BERT encoder and obtain final hidden layer embeddings
    output: bertEmbedding - BERT embeddings for a sentence 
    '''
    def GetSentenceEmbedding(self, tokensTensor, sentenceTensor):
        with torch.no_grad():
            encoderOutput = self.model(tokensTensor, sentenceTensor)

            finalHiddenState = encoderOutput[2][-1][0]
            
        return finalHiddenState
    
    '''
    param: groupedEmbeddingList - a list of embeddings of the same size to be averaged
    purpose: combine a list of embeddings via mean/average
    output: combined embeddings
    '''    
    def AveragingFunction(self, groupedEmbeddingList):
        return torch.mean(groupedEmbeddingList, 0)

    '''
    param: bertEmbeddingOutput - BERT embeddings for a sentence
    param: wpMappingDict - dictionary of vectors of indices to allow the reassembly of wordpiece tokens and 
        embeddings
    param: averagingFunction - the function used to combine embeddings
    purpose: reassemble word piece into sentences. reassemble embeddings by combining embeddings
        for word pieces that belong to the same word
    output: bertEmbedding - resassembled BERT embedding
    '''
    def ReassembleEmbedding(self, bertEmbeddingOutput, tokenizedSentence):
        if bertEmbeddingOutput.shape[0] != len(tokenizedSentence):
            raise Exception("BERT embeddings output vector does not match tokenized sentence in length")
        
        groupedWpEmbeddings = []
        for token, embedding in zip(tokenizedSentence, bertEmbeddingOutput[:, None]):
            if '##' in token:
                groupedWpEmbeddings[-1] = torch.concat((groupedWpEmbeddings[-1], embedding))
            else:
                groupedWpEmbeddings.append(embedding)

        bertEmbeddings = torch.tensor([])
        for embeddingSet in groupedWpEmbeddings:
            t1 = self.AveragingFunction(torch.tensor(embeddingSet))

            bertEmbeddings = torch.concat((bertEmbeddings, t1[None,:]))

        return bertEmbeddings