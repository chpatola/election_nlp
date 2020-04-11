import pandas as pd
import numpy as np
import nltk #library for word processing
import string
import re # library for regular expressions
from nltk.tokenize import RegexpTokenizer # divide sentences to words/signs/patterns
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
#lemmmatization https://antupis.github.io/lemmatization/finnish/2019/06/12/Lemmatizing-finnish-text.html
from src import Generic_func as gf
from sklearn.feature_extraction.text import CountVectorizer#bag of words
from collections import Counter
import scattertext as st
import spacy
nlp =spacy.load('en_core_web_sm') 

#https://towardsdatascience.com/a-complete-exploratory-data-analysis-and-visualization-for-text-data-29fb1b96fb6a
#https://towardsdatascience.com/nlp-for-beginners-cleaning-preprocessing-text-data-ae8e306bef0f
#https://medium.com/@rrfd/cookiecutter-data-science-organize-your-projects-atom-and-jupyter-2be7862f487e


def simplify_sentences(text, stop ="no"):
    tokenizer = RegexpTokenizer('\w+|\$[\d\.]+|\S+') #splits up by spaces or by periods that are not attached to a digit
    stemmer = SnowballStemmer("finnish")

    newtext = "".join([sign for sign in text if sign not in string.punctuation])
    newtext =tokenizer.tokenize(newtext.lower())
    newtext  = " ".join([stemmer.stem(word) for word in newtext ]) #As is it last step, we join the words back to a sentence
    return newtext

def wordfreqdf(df):
    corpus = st.CorpusFromPandas(yle_data, category_col='party', text_col='work_for', nlp=nlp).build()
    term_freq_df = corpus.get_term_freq_df()

    result_df = pd.DataFrame(columns=["1","2","3","4","5","6"])
    parties= df['party'].sort_values().unique()

    index =0
    for party in parties:
        party_score_name = party + "_Score"
        term_freq_df[party_score_name] = corpus.get_scaled_f_scores(party)
        result_df.loc[index] = list(term_freq_df.sort_values(by=party_score_name, ascending=False).index[:6])
        index = index +1
    
    result_df["Party"] = parties
    result_df.set_index("Party",inplace=True)

    return result_df  

#1. Import data 
path = '/home/chpatola/Desktop/Skola/Python/cookie_nlp/data/interim/cleaned_data.csv'
yle_data = pd.read_csv(path, sep=',', encoding="ISO-8859-1")

#2. Preprocess sentences
yle_data["work_for"] = yle_data["work_for"].apply(lambda x : simplify_sentences(x))
yle_data.head(5)

# 3. Write processed data to folder
yle_data.to_csv("/home/chpatola/Desktop/Skola/Python/cookie_nlp/data/processed/processed_data.csv",index=False,encoding="ISO-8859-1")

#4. Get insights into 6 most typical words for each party
party_word_freq = wordfreqdf(yle_data)
party_word_freq.to_csv('/home/chpatola/Desktop/Skola/Python/cookie_nlp/reports/party_word_freq.csv',sep=',', encoding="ISO-8859-1")

