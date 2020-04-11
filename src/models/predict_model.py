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
val_X, val_y, bag, tf_idf = train_model.train_model()

print("Mean accurancy: {:.2f} %".format(100*bag.best_score_))

predictions =bag.predict(val_X)
print("Predictions:\n {} \nTruth:\n {}".format(predictions[0:3],val_y[0:3]))
print(val_X[0:3])

#2. Save classification report and confusion matrix to file
classi_rep = Visualize._plot_classification_report(val_y,predictions)
classi_rep.savefig('/home/chpatola/Desktop/Skola/Python/cookie_nlp/reports/figures/classificationReport.png',bbox_inches='tight')

parties= val_y.sort_values().unique()
Visualize.cm_analysis(val_y,predictions,'/home/chpatola/Desktop/Skola/Python/cookie_nlp/reports/figures/confusion_matrix.png',labels=parties)
print(confusion_matrix(val_y,predictions))
print(classification_report(val_y,predictions))
print(accuracy_score(val_y,predictions))

