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

#1. Test model and look into results
val_X, val_y, bag, tf_idf = train_model.train_model()

predictions =bag.predict(val_X)
print("Predictions:\n {} \nTruth:\n {}".format(predictions[0:3],val_y[0:3]))
print(val_X[0:3])

print("Mean accurancy: {:.2f} %".format(100*bag.best_score_))
print(classification_report(val_y,predictions))
print("Row is truth, columns guess\n {}".format(confusion_matrix(val_y,predictions)))

#To check: 10 most common words in each group, % chans for each party for a specific guess
#+ some nice visualizations