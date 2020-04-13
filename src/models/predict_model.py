import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from src.models import train_model
from lime import lime_text
from lime.lime_text import LimeTextExplainer
import matplotlib.pyplot as plt
import itertools
from src.visualization import Visualize
import seaborn as sns
from sklearn.metrics import accuracy_score

#1. Test model and look into results
test_X, test_y, bag, tf_idf = train_model.train_model()

print("Mean accurancy in validation: {:.2f} %".format(100*bag.best_score_))

predictions =bag.predict(test_X)
print("Predictions:\n {} \nTruth:\n {}".format(predictions[0:3],test_y[0:3]))
print(test_X[0:3])

#2. Save classification report and confusion matrix to file
classi_rep = Visualize._plot_classification_report(test_y,predictions)
classi_rep.savefig('/home/chpatola/Desktop/Skola/Python/cookie_nlp/reports/figures/classificationReport.png',bbox_inches='tight')

parties= test_y.sort_values().unique()
Visualize.cm_analysis(test_y,predictions,'/home/chpatola/Desktop/Skola/Python/cookie_nlp/reports/figures/confusion_matrix.png',labels=parties)
print(confusion_matrix(test_y,predictions))
print(classification_report(test_y,predictions))
print("Accurancy in test: {:.2f} %".format(100*(accuracy_score(test_y,predictions))))

