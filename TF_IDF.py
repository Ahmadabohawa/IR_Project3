import joblib
from spacy.lang.en import English
import pandas as pd

nlp = English()
def lemmatization(token_txt):
    doc = nlp(" ".join(token_txt))
    lemmas = [token.lemma_ if token.pos_ in ["ADJ", "ADV", "NOUN", "VERB"] else token.text for token in doc]
    return lemmas

    #----------------------------------Wikir Dataset ---------------------------#
# load the data from the file using joblib
wiki = joblib.load('C:/Users/USER/PycharmProjects/FinalIR/wikiData/wiki.joblib')

wiki_query = joblib.load('C:/Users/USER/PycharmProjects/FinalIR/wikiData/queries_processed/wiki_qry_processed.joblib')

# TFIDF for documents of wiki_dataset
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(analyzer=lemmatization)
documents_vector = tfidf.fit_transform(wiki)
print(documents_vector.shape)
df = pd.DataFrame(documents_vector.toarray(),columns = tfidf.get_feature_names_out())


joblib.dump(tfidf,'C:/Users/USER/PycharmProjects/FinalIR/wikiData/tfidf.joblib')
joblib.dump(documents_vector,'C:/Users/USER/PycharmProjects/FinalIR/wikiData/documents_vector.joblib')


# TFIDF for queries of wiki_queries
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_query = TfidfVectorizer(analyzer=lemmatization)
queries_vector = tfidf_query.fit_transform(wiki_query["text_lemmatized"])
print(queries_vector.shape)
df = pd.DataFrame(queries_vector.toarray(),columns = tfidf_query.get_feature_names_out())

joblib.dump(tfidf_query,'C:/Users/USER/PycharmProjects/FinalIR/wikiData/queries_processed/tfidf_query.joblib')
joblib.dump(queries_vector,'C:/Users/USER/PycharmProjects/FinalIR/wikiData/queries_processed/queries_vector.joblib')

# ----------------------------------Antique Dataset ---------------------------
# load the data_lemmatized from the file antique using joblib
antique = joblib.load('C:/Users/USER/PycharmProjects/FinalIR/AntiqueData/antique.joblib')

antique_query = joblib.load('C:/Users/USER/PycharmProjects/FinalIR/AntiqueData/Queries_processed/antique_qry_processed.joblib')

# TFIDF for documents of antique_dataset
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(analyzer=lemmatization)
documents_vector_antique = tfidf.fit_transform(antique)
print(documents_vector_antique.shape)
# df_antique_documents = pd.DataFrame(documents_vector_antique.toarray(),columns = tfidf.get_feature_names_out())


joblib.dump(tfidf,'C:/Users/USER/PycharmProjects/FinalIR/AntiqueData/tfidf_antique.joblib')
joblib.dump(documents_vector_antique,'C:/Users/USER/PycharmProjects/FinalIR/AntiqueData/documents_vector_antique.joblib')


# TFIDF for queries of antique_queries
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_query_antique = TfidfVectorizer(analyzer=lemmatization)
queries_vector_antique = tfidf_query_antique.fit_transform(antique_query["text_lemmatized"])
print(queries_vector_antique.shape)
df_antique_query = pd.DataFrame(queries_vector_antique.toarray(),columns = tfidf_query_antique.get_feature_names_out())

joblib.dump(tfidf_query_antique,'C:/Users/USER/PycharmProjects/FinalIR/AntiqueData/Queries_processed/tfidf_query_antique.joblib')
joblib.dump(queries_vector_antique,'C:/Users/USER/PycharmProjects/FinalIR/AntiqueData/Queries_processed/queries_vector_antique.joblib')

