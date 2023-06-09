# ------------------------ Wikir Query --------------------------
import nltk
import pandas as pd
import joblib
import re
import string
ps = nltk.PorterStemmer()
stop_words = nltk.corpus.stopwords.words('english')

def remove_punctuation(txt):
    txt_nopuct = "".join([c for c in txt if c not in string.punctuation])
    return txt_nopuct


def reg_ex(text):
    text = re.sub(r'\b(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})\b', r'\3-\1-\2', text)

    # remove Abbreviations
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"n't", "not", text)
    text = re.sub(r"'re", "are", text)
    text = re.sub(r"'s", "is", text)
    text = re.sub(r"'d", "would", text)
    text = re.sub(r"'ll", "will", text)
    text = re.sub(r"'t", "not", text)
    text = re.sub(r"'ve", "have", text)

    # Check from numbers and letters exist
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)

    return text


def tokenize(txt):
    # 1- tokens = txt.split(" ")
    # 2-  tokens = re.split('\W+',txt)
    tokens = nltk.word_tokenize(txt)

    return tokens


def remove_stopwords(txt_tokenized):
    txt_clean = [word for word in txt_tokenized if word not in stop_words]
    return txt_clean

def stemming(tokenized_text):
    text = [ps.stem(word) for word in tokenized_text]
    return text


#import spacy library and download english language from spacy.lang.en import English.
import spacy
from spacy.lang.en import English
#create an object with nlp by using English() to use english model which exist in spacy
#use nlp to convert (token_txt) to Doc

nlp = English()
def lemmatization(token_txt):
    # token.lemma_ : to convert every word to its main shape
    # token.pos_: to check that the type of the main word ? [adj , noun ...]
    # and otherwise we will use the original text for the word

    doc = nlp(" ".join(token_txt))
    lemmas = [token.lemma_ if token.pos_ in ["ADJ", "ADV", "NOUN", "VERB"] else token.text for token in doc]
    return lemmas


# Wikir_query=pd.read_csv("Wikir1k/queries.csv")
# Wikir_query.columns=['id','text']
#
# Wikir_query['text_clean']=Wikir_query["text"].apply(lambda x:remove_punctuation(x))
# print("punctuation")
# Wikir_query['text_regex']=Wikir_query["text_clean"].apply(lambda x:reg_ex(x))
# print("regex")
# Wikir_query['text_clean_tokenize']= Wikir_query['text_regex'].apply(lambda x :tokenize(x.lower()))
# Wikir_query['txt_no_SW']= Wikir_query['text_clean_tokenize'].apply(lambda x :remove_stopwords(x))
# Wikir_query['text_steemed']= Wikir_query['txt_no_SW'].apply(lambda x : stemming(x))
# print("steemed")
# Wikir_query['text_lemmatized']= Wikir_query['text_steemed'].apply(lambda x:lemmatization(x))
# print("lemmatized")
# #Convert it to String
# Wikir_query['text_lemmatized_str'] = Wikir_query['text_lemmatized'].apply(lambda x: ' '.join(x))


# joblib.dump(Wikir_query, 'C:/Users/USER/PycharmProjects/FinalIR/wikiData/queries_processed/wiki_qry_processed.joblib')

# ------------------------------- Antique query --------------------------------
# Read the TSV file into a
antique_query = pd.read_csv('Antique/antique_queries.tsv', sep='\t')

antique_query['text_clean']=antique_query["text"].apply(lambda x:remove_punctuation(x))
print("punctuation")
antique_query['text_regex']=antique_query["text_clean"].apply(lambda x:reg_ex(x))
print("regex")
antique_query['text_clean_tokenize']= antique_query['text_regex'].apply(lambda x :tokenize(x.lower()))
antique_query['txt_no_SW']= antique_query['text_clean_tokenize'].apply(lambda x :remove_stopwords(x))
antique_query['text_steemed']= antique_query['txt_no_SW'].apply(lambda x : stemming(x))
print("steemed")
antique_query['text_lemmatized']= antique_query['text_steemed'].apply(lambda x:lemmatization(x))
print("lemmatized")
#Convert it to String
antique_query['text_lemmatized_str'] = antique_query['text_lemmatized'].apply(lambda x: ' '.join(x))

joblib.dump(antique_query, 'C:/Users/USER/PycharmProjects/FinalIR/AntiqueData/Queries_processed/antique_qry_processed.joblib')
