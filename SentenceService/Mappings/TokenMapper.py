from SentenceService.Models.RegisterTokenModel import RegisterTokenModel

def MapToRegisterTokenModel(tokenId, token, pos, stem, sentIndex, sentEmbeddingId):
    model = RegisterTokenModel()

    model.Id, model.Value, model.Pos, model.Stem = tokenId, token, pos, stem
    
    model.SentEmbeddingIndex, model.SentEmbeddingId = sentIndex, sentEmbeddingId
    
    return model