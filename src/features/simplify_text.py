import pandas as pd
import numpy as np
import nltk #library for word processing
import string
import re # library for regular expressions
from nltk.tokenize import RegexpTokenizer # divide sentences to words/signs/patterns
from nltk.stem.snowball import SnowballStemmer
#lemmmatization https://antupis.github.io/lemmatization/finnish/2019/06/12/Lemmatizing-finnish-text.html
from src import Generic_func as gf
from sklearn.feature_extraction.text import CountVectorizer

#https://towardsdatascience.com/nlp-for-beginners-cleaning-preprocessing-text-data-ae8e306bef0f
#https://medium.com/@rrfd/cookiecutter-data-science-organize-your-projects-atom-and-jupyter-2be7862f487e


def prepare_sentences(text):
    tokenizer = RegexpTokenizer('\w+|\$[\d\.]+|\S+') #splits up by spaces or by periods that are not attached to a digit
    stemmer = SnowballStemmer("finnish")

    newtext = "".join([sign for sign in text if sign not in string.punctuation])
    newtext =tokenizer.tokenize(newtext.lower())
    newtext  = " ".join([stemmer.stem(word) for word in newtext ])#As is it last step, we join the words back to a sentence
    return newtext

#1. Import data and correct scandinavian characters
path = '/home/chpatola/Desktop/Skola/Python/cookie_nlp/data/interim/cleaned_data.csv'
yle_data = pd.read_csv(path, sep=',', encoding="ISO-8859-1")

#2. Preprocess sentences
yle_data["work_for"] = yle_data["work_for"].apply(lambda x : prepare_sentences(x))
yle_data.head(5)

# 3. Write processed data to folder
yle_data.to_csv("/home/chpatola/Desktop/Skola/Python/cookie_nlp/data/processed/processed_data.csv",index=False,encoding="ISO-8859-1")


