

class TokenRegistrar:
    '''
    class: TokenRegistrar
    purpose: call functions to create POS vector, token vector and apply model to create 
        embedding vector. Generate code to house token - embedding - sentence database for lookups
    params: the BERTModelWrapper
    '''
    def __init__(self, WrapperModel):
        return
    
    '''
    param: sentence - a tokenized plaintext sentence
    purpose: part-of-speech(POS) tag input sentence
    output: posVector - a vector of POS tags equal in length to the tokenized sentence
    '''
    def TagPOS(sentence):
        return
    
    '''
    param: sentence - a tokenized plaintext sentence
    purpose: fix word spellings and run other preprocessing tools on sentence
    output: correctedSentence - plaintext proprocessing corrected sentence
    '''
    def PreProcessSentence(sentence):
        return

    '''
    param: sentence - a plaintext sentence
    purpose: pass sentence through the model to get embedding
    output: embeddings - embeddings obtained from the final output of model
    '''
    def ModelGetSentenceEmbedding(sentence):
        return
    
    '''
    param: tableName - table to write data to
    param: embedding - embedding to insert into table
    purpose: create the file reference and write the code that creates the table
    sql table
    output: success / failure
    '''
    def RegisterSQLTable(tableName, embedding):
        return
    
    '''
    param: tableName - table to make insertions to
    param: insertItem - item to be inserted into table
    purpose: write the DDL to insert row into table
    output: success / failure
    '''    
    def InsertToTable(tableName, insertItem):
        return

    '''
    param: embedding - embedding to insert into table
    param: sentence - sentence to insert into table
    purpose: for each embedding, generate code to insert embedding into table with auto 
        increment id. For each sentence, generate code to insert embedding into table with 
        auto increment id. For each token, insert embedding into table with the ids 
        shared by sentence & embedding
    '''
    def TokenContextToDB(embedding, sentence):
        return


