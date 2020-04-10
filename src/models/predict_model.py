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

#1. Test model and look into results
val_X, val_y, bag, tf_idf = train_model.train_model()

predictions =bag.predict(val_X)
print("Predictions:\n {} \nTruth:\n {}".format(predictions[0:3],val_y[0:3]))
print(val_X[0:3])

print("Mean accurancy: {:.2f} %".format(100*bag.best_score_))
print(classification_report(val_y,predictions))
print("Row is truth, columns guess\n {}".format(confusion_matrix(val_y,predictions)))

explainer = LimeTextExplainer(class_names=np.unique(val_y))
exp = explainer.explain_instance(val_X[0], bag.predict_proba, num_features=6, labels=[0, 1])
print ('Explanation for class %s' % np.unique(val_y)[0])
print ('\n'.join(map(str, exp.as_list(label=0))))
exp.show_in_notebook(text=False)
exp.show_in_notebook(text=val_X[0], labels=(0,))
#% chans for each party for a specific row
print("Probability for text to belong to each part:\n{}".format(bag.predict_proba([val_X[0]]).round(3)*100))

