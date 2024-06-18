from Helpers import SentenceHelpers
from SentenceService.Mappings import TokenMapper
from SentenceService.Mappings import EmbeddingMapper

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
        self.currentTokenId = 0
        self.currentEmbeddingId = 0
        return
    
    '''
    param: sentence - a list of words from a sentence split by whitespace
    purpose: fix word spellings and run other preprocessing tools on sentence
    output: correctedSentence - plaintext proprocessing corrected sentence
    '''
    def SurveySentence(self, sentence):
        stemmer = PorterStemmer()
        
        pos = [tp[1] for tp in nltk.pos_tag(sentence)]
        isStopWord = [token in stopwords.words('english') for token in sentence]
        stems = [stemmer.stem(token) for token in sentence]

        return pos, isStopWord, stems

    '''
    param: sentence - a plaintext sentence
    purpose: pass sentence through the model to get embedding
    output: embeddings - embeddings obtained from the final output of model
    '''
    def ModelGetSentenceEmbedding(self, sentence):

        tokensTensor, tokenizedSentence = self.embeddingsModel.TokenizeSentence(sentence)
        sentenceTensor = self.embeddingsModel.GetSentenceIdVector(tokensTensor)
        bertEmbedding = self.embeddingsModel.GetSentenceEmbedding(tokensTensor, sentenceTensor)
        
        return self.embeddingsModel.ReassembleEmbedding(bertEmbedding, tokenizedSentence)
    '''
    param: sentence - a plaintext sentence
    purpose: register to db the tokens and embeddings from a sentence
    output: None
    '''
    def RegisterSentence(self, sentence):

        if sentence == "" or sentence == None:
            return

        sentenceSplit = SentenceHelpers.sentenceSplitWhitespace(sentence)
        sentenceLen = len(sentenceSplit)

        sentPos, isStopWordVector, stems = self.SurveySentence(sentenceSplit)
        embeddings = self.ModelGetSentenceEmbedding(sentence)

        if len(sentPos) != sentenceLen:
            Exception("Length of pos vector does not match sentence length")
        if len(isStopWordVector) != sentenceLen:
            Exception("Length of stop words vector does not match sentence length")
        if len(stems) != sentenceLen:
            Exception("Length of stems does not match sentence length")
        if embeddings.shape[0] != sentenceLen + 2:
            Exception("Length of embeddings does not match length of tokens")

        embeddingModel = EmbeddingMapper.MapToSentenceEmbeddingsModel(self._GetNextAvailableSentenceId(), sentenceSplit, embeddings)
        self.currentEmbeddingId += 1

        embeddingModelList = []
        for token, pos, isStopWord, stem, index in zip(sentenceSplit, sentPos, isStopWordVector, stems, list(range(0,sentenceLen))):
            if isStopWord :
                continue
            else:
                tokenModel = TokenMapper.MapToRegisterTokenModel(self._GetNextAvailableTokenId(), token, pos, stem, 
                                                    index, self.currentEmbeddingId)
                
                self.currentTokenId += 1

                embeddingModelList.append(tokenModel)

        return embeddingModel, embeddingModelList
    
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

    def _GetNextAvailableTokenId(self):
        return self.currentTokenId + 1
    
    def _GetNextAvailableSentenceId(self):
        return self.currentTokenId + 1