from TokenRegistrar.Models.RegisterTokenModel import RegisterTokenModel

def MapToRegistarTokenModel(token, pos, stem, sentIndex, sentId):
    model = RegisterTokenModel

    model.Value, model.Pos, model.Stem = token, pos, stem
    
    model.SentEmbeddingIndex, model.SentEmbeddingId = sentIndex, sentId
    
    return model