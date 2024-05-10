class Config:
    
    # repo path
    path = "C:/Users/kdzhou/source/repos/VedaOntoSentAnalysis"

    # data path
    data_path = f"{path}/data"
    raw_data_path = f"{data_path}/loneliness_tweets_t6.csv"
    topic_grouped_data_path = ""  

    # lexcal sources path
    lexical_sources = f"{path}/lexical_sources"
    nltk_lexical_sources = [
        'gutenberg',#98552 ##2621613
    ]