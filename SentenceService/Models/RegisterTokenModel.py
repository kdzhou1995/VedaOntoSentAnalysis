class RegisterTokenModel(object):

    def __init__(self):
        self.Id = None
        self.Value = None
        self.Pos = None
        self.Stem = None
        self.SentEmbeddingIndex = None
        self.SentEmbeddingId = None

    def __repr__(self):
        s1 = f"Id:{self.Id}\nValue:{self.Value}\n\tPos:{self.Pos}\n\tStem:{self.Stem}"
        return s1 + f"\n\tSentence Embedding Index:{self.SentEmbeddingIndex}\n\tSentence Embedding Id:{self.SentEmbeddingId}"