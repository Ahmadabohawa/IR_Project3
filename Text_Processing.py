import nltk
import pandas as pd
import string # for punctuation
import re # for tokenization
from nltk.stem import PorterStemmer # for Stemming
ps = nltk.PorterStemmer()
stop_words = nltk.corpus.stopwords.words('english')
lm = nltk.WordNetLemmatizer()
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import joblib
pd.options.mode.chained_assignment = None

# # wiki=pd.read_csv("Wikir1k/documents.csv")
# # wiki.columns=['id','text']
#
antique = pd.read_csv("Antique/antique2.tsv",sep='\t')
antique.columns=["id","text"]

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
    text = re.sub(r"U.S.A", "United State American", text)

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


ps = PorterStemmer()


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


antique['text_clean']=antique["text"].apply(lambda x:remove_punctuation(x))
print("punctuation")
antique['text_regex']=antique["text_clean"].apply(lambda x:reg_ex(x))
antique['text_clean_tokenize']= antique['text_regex'].apply(lambda x :tokenize(x.lower()))
antique['txt_no_SW']= antique['text_clean_tokenize'].apply(lambda x :remove_stopwords(x))
antique['text_steemed']= antique['txt_no_SW'].apply(lambda x : stemming(x))
print("steemed")
antique['txt_lemmatized']= antique['text_steemed'].apply(lambda x:lemmatization(x))

joblib.dump(antique["txt_lemmatized"], 'C:/Users/USER/PycharmProjects/FinalIR/AntiqueData/antique.joblib')
doc_id_list_antique = antique['id']
joblib.dump(doc_id_list_antique, 'C:/Users/USER/PycharmProjects/FinalIR/AntiqueData/doc_id_list_antique.joblib')

# # Create inverted index

from collections import defaultdict

# Create a defaultdict to store the inverted index
inverted_index_antique = defaultdict(list)

# Iterate over each document and each term in the document
for doc_id, doc in enumerate(antique['txt_lemmatized']):
     for term in doc:
         # Add the document ID to the list of document IDs associated with the term
         inverted_index_antique[term].append(doc_id)
# # Print the inverted index
# # for term, doc_ids in inverted_index.items():
# #     print(term, doc_ids)
#
joblib.dump(inverted_index_antique, 'C:/Users/USER/PycharmProjects/FinalIR/AntiqueData/inverted_index_antique.joblib')


# wiki['text_clean']=wiki["text"].apply(lambda x:remove_punctuation(x))
# wiki['text_regex']=wiki["text_clean"].apply(lambda x:reg_ex(x))
# wiki['text_clean_tokenize']= wiki['text_regex'].apply(lambda x :tokenize(x.lower()))
# wiki['txt_no_SW']= wiki['text_clean_tokenize'].apply(lambda x :remove_stopwords(x))
# wiki['text_steemed']= wiki['txt_no_SW'].apply(lambda x : stemming(x))
# wiki['txt_lemmatized']= wiki['text_steemed'].apply(lambda x:lemmatization(x))
#
# # Create inverted index
#
# from collections import defaultdict
#
# # Create a defaultdict to store the inverted index
# inverted_index = defaultdict(list)
#
# # Iterate over each document and each term in the document
# for doc_id, doc in enumerate(wiki['txt_lemmatized']):
#     for term in doc:
#         # Add the document ID to the list of document IDs associated with the term
#         inverted_index[term].append(doc_id)
#
#
# # Print the inverted index
# # for term, doc_ids in inverted_index.items():
# #     print(term, doc_ids)
#
# joblib.dump(wiki["txt_lemmatized"], 'C:/Users/USER/PycharmProjects/FinalIR/wikiData/wiki.joblib')
# doc_id_list = wiki['id']
# joblib.dump(doc_id_list, 'C:/Users/USER/PycharmProjects/FinalIR/wikiData/doc_id_list.joblib')
# joblib.dump(inverted_index, 'C:/Users/USER/PycharmProjects/FinalIR/wikiData/inverted_index.joblib')
#
#
