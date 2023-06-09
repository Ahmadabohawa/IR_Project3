from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import joblib
from spacy.lang.en import English

nlp = English()
def lemmatization(token_txt):
    doc = nlp(" ".join(token_txt))
    lemmas = [token.lemma_ if token.pos_ in ["ADJ", "ADV", "NOUN", "VERB"] else token.text for token in doc]
    return lemmas

wiki_query_documents = {}

wiki = joblib.load('C:/Users/USER/PycharmProjects/FinalIR/wikiData/wiki.joblib')

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(analyzer=lemmatization)
documents_vector_wiki = tfidf.fit_transform(wiki)


wiki_query = joblib.load('C:/Users/USER/PycharmProjects/FinalIR/wikiData/queries_processed/wiki_qry_processed.joblib')
documents_vector = joblib.load('C:/Users/USER/PycharmProjects/FinalIR/wikiData/documents_vector.joblib')
documents_id = joblib.load('C:/Users/USER/PycharmProjects/FinalIR/wikiData/doc_id_list.joblib')


# Note : read the down note
def Cosine_Similarity(wiki_query,tfidf,documents_vector,documents_id):
    for query_id in range(len(wiki_query['text_lemmatized_str'])):
        query_vector = tfidf.transform([wiki_query["text_lemmatized_str"][query_id]])
        similarities = cosine_similarity(query_vector.toarray().flatten().reshape(1, -1), documents_vector)
        sorted_indices = similarities.argsort()[::-1]
        top_10_indices = sorted_indices[:10]
        retrieved_docs = [documents_id[index] for index in top_10_indices]
        wiki_query_documents[str(wiki_query["id"][query_id])] = retrieved_docs

    return wiki_query_documents


print(Cosine_Similarity(wiki_query,tfidf,documents_vector_wiki,documents_id))

# Note 1- : the output of this function will be  dictionary which has every single query with its top 10 files
# relevance  , So we can use it with qrels dictionary to calculate the evaluation

# 2- : for the short time we used the wikir DataSet to calculate Cosine_Similarity
