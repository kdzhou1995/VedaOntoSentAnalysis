from Helpers import SentenceHelpers
from SentenceService.Mappings import TokenMapper

import nltk
from nltk.corpus import stopwords
from nltk.stem import *

class SentenceService(object):
    '''
    class: SentenceService
    purpose: call functions to create POS vector, token vector and apply model to create 
        embedding vector. Generate code to house token - embedding - sentence database for lookups
    params: the BERTModelWrapper
    '''
    def __init__(self, WrapperModel):
        self.embeddingsModel = WrapperModel
        return
    
    '''
    param: sentence - a list of words from a sentence split by whitespace
    purpose: fix word spellings and run other preprocessing tools on sentence
    output: correctedSentence - plaintext proprocessing corrected sentence
    '''
    def SurveySentence(self, sentence):
        stemmer = PorterStemmer()
        
        pos = [tp[1] for tp in nltk.pos_tag(sentence)]
        isStopWord = [token in stopwords for token in sentence]
        stems = [stemmer.stem(token) for token in sentence]

        return pos, isStopWord, stems
    
    '''
    param: sentence - a plaintext sentence
    purpose: pass sentence through the model to get embedding
    output: embeddings - embeddings obtained from the final output of model
    '''
    def ModelGetSentenceEmbedding(self, sentence):

        tokensTensor, tokenizedSentence = self.embeddingsModel.TokenizeSentence(sentence)
        sentenceTensor = self.GetSentenceIdVector(tokensTensor)
        bertEmbedding = self.embeddingsModel.GetSentenceEmbedding(tokensTensor, sentenceTensor)

        sentenceLen = SentenceHelpers.sentenceSplitWhitespace(sentence)
        
        return self.embeddingsModel.ReassembleEmbedding(bertEmbedding, tokenizedSentence, sentenceLen)
    '''
    param: sentence - a plaintext sentence
    param: pos - a part of speech vector of each token in sentence
    param: isStopWord - a bit indicating if token is
    '''
    def RegisterTokensEmbeddings(self, sentence):

        sentenceSplit = SentenceHelpers.sentenceSplitWhitespace(sentence)
        sentenceLen = len(sentenceSplit)

        pos, isStopWord, stems = self.SurveySentence(sentenceSplit)
        embeddings = self.ModelGetSentenceEmbedding(sentence)

        if len(pos) != sentenceLen:
            Exception("Length of pos vector does not match sentence length")
        if len(isStopWord) != sentenceLen:
            Exception("Length of stop words vector does not match sentence length")
        if len(stems) != sentenceLen:
            Exception("Length of stems does not match sentence length")
        if len(embeddings) != sentenceLen:
            Exception("Length of ")

        TokenMapper.MapToRegistarTokenModel

        return
    
    '''
    param: tableName - table to write data to
    param: embedding - embedding to insert into table
    purpose: create the file reference and write the code that creates the table
    sql table
    output: success / failure
    '''
    def RegisterSQLTable(self, tableName, embedding):
        return
    
    '''
    param: tableName - table to make insertions to
    param: insertItem - item to be inserted into table
    purpose: write the DDL to insert row into table
    output: success / failure
    '''
    def GenerateInsertToTable(self, tableName, registerTokenModel):
        return

    '''
    param: embedding - embedding to insert into table
    param: sentence - sentence to insert into table
    purpose: for each embedding, generate code to insert embedding into table with auto 
        increment id. For each sentence, generate code to insert embedding into table with 
        auto increment id. For each token, insert embedding into table with the ids 
        shared by sentence & embedding
    '''
    def GenerateTokenContextToDB(self, embedding, sentence):
        return
