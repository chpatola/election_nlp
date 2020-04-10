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
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
import scattertext as st
import spacy
nlp =spacy.load('en_core_web_sm') 

#https://towardsdatascience.com/a-complete-exploratory-data-analysis-and-visualization-for-text-data-29fb1b96fb6a
#https://towardsdatascience.com/nlp-for-beginners-cleaning-preprocessing-text-data-ae8e306bef0f
#https://medium.com/@rrfd/cookiecutter-data-science-organize-your-projects-atom-and-jupyter-2be7862f487e


def prepare_sentences(text, stop ="no"):
    tokenizer = RegexpTokenizer('\w+|\$[\d\.]+|\S+') #splits up by spaces or by periods that are not attached to a digit
    stemmer = SnowballStemmer("finnish")

    newtext = "".join([sign for sign in text if sign not in string.punctuation])
    newtext =tokenizer.tokenize(newtext.lower())
    newtext  = " ".join([stemmer.stem(word) for word in newtext ]) #As is it last step, we join the words back to a sentence
    return newtext

#1. Import data and correct scandinavian characters
path = '/home/chpatola/Desktop/Skola/Python/cookie_nlp/data/interim/cleaned_data.csv'
yle_data = pd.read_csv(path, sep=',', encoding="ISO-8859-1")
yle_data_explore = yle_data.copy()

#2. Preprocess sentences
yle_data["work_for"] = yle_data["work_for"].apply(lambda x : prepare_sentences(x))
yle_data.head(5)

# 3. Write processed data to folder
yle_data.to_csv("/home/chpatola/Desktop/Skola/Python/cookie_nlp/data/processed/processed_data.csv",index=False,encoding="ISO-8859-1")



#4. Get insights into 10 most common word for each party
corpus = st.CorpusFromPandas(yle_data, category_col='party', text_col='work_for', nlp=nlp).build()
term_freq_df = corpus.get_term_freq_df()

term_freq_df['Kokoomus_score'] = corpus.get_scaled_f_scores('Kansallinen Kokoomus')
print(list(term_freq_df.sort_values(by='Kokoomus_score', ascending=False).index[:10]))
term_freq_df['Perus_score'] = corpus.get_scaled_f_scores('Perussuomalaiset')
print(list(term_freq_df.sort_values(by='Perus_score', ascending=False).index[:10]))
term_freq_df['Keskusta_score'] = corpus.get_scaled_f_scores('Suomen Keskusta')
print(list(term_freq_df.sort_values(by='Keskusta_score', ascending=False).index[:10]))
term_freq_df['Krist_score'] = corpus.get_scaled_f_scores('Suomen Kristillisdemokraatit (KD)')
print(list(term_freq_df.sort_values(by='Krist_score', ascending=False).index[:10]))
term_freq_df['SDP_score'] = corpus.get_scaled_f_scores('Suomen Sosialidemokraattinen Puolue')
print(list(term_freq_df.sort_values(by='SDP_score', ascending=False).index[:10]))
term_freq_df['Vasen_score'] = corpus.get_scaled_f_scores('Vasemmistoliitto')
print(list(term_freq_df.sort_values(by='Vasen_score', ascending=False).index[:10]))
term_freq_df['Vihreä_score'] = corpus.get_scaled_f_scores('Vihreä liitto')
print(list(term_freq_df.sort_values(by='Vihreä_score', ascending=False).index[:10]))




yle_data_explore["work_for"] = yle_data_explore["work_for"].apply(lambda x : x.lower())
yle_data_explore["work_for"] = yle_data_explore["work_for"].apply(lambda x : "".join([sign for sign in x if sign not in string.punctuation]))  
#yle_data_explore["work_for"] = yle_data_explore["work_for"].apply(lambda x : "".join([sign for sign in x if sign not in stopwords.words("finnish")])) 
yle_data_explore.head(4)
yle_data_explore["work_for"][0][0]

kokoomus_most_common = Counter(" ".join(yle_data_explore[yle_data_explore["party"]=="Kansallinen Kokoomus"].work_for.dropna()).split()).most_common(10)
print(kokoomus_most_common)
perus_most_common = Counter(" ".join(yle_data_explore[yle_data_explore["party"]=="Perussuomalaiset"].work_for.dropna()).split()).most_common(10)
print(perus_most_common)
keskusta_most_common = Counter(" ".join(yle_data_explore[yle_data_explore["party"]=="Suomen Keskusta"].work_for.dropna()).split()).most_common(10)
print(keskusta_most_common)