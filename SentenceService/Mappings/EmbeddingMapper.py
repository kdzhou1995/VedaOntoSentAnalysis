from SentenceService.Models.RegisterEmbeddingModel import RegisterEmbeddingModel

def MapToSentenceEmbeddingsModel(sentenceId, tokens, embeddings):
    model = RegisterEmbeddingModel()

    model.Id = sentenceId
    model.Tokens = tokens
    model.Embeddings = embeddings

    return model