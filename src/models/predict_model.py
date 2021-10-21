"""Use prediction model and evaluate it"""
from os import path
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from src.models import train_model
from src.visualizations import Visualize

base_path ='/home/chpatola/Desktop/Skola/Python/cookie_nlp/'

#1. Test model and look into results
test_X, test_y, bag, tf_idf = train_model.train_model(base_path)

print("Mean accurancy in validation: {:.2f} %".format(100*bag.best_score_))

predictions = bag.predict(test_X)
print("Predictions:\n {} \nTruth:\n {}".format(predictions[0:3], test_y[0:3]))
print(test_X[0:3])

#2. Save classification report and confusion matrix to file
classi_rep = Visualize._plot_classification_report(test_y, predictions)
classi_rep.savefig(
    path.join(base_path,'reports/figures/classificationReport.png'),
    bbox_inches='tight')
parties = test_y.sort_values().unique()
Visualize.cm_analysis(test_y,
                      predictions,
                      path.join(base_path,'reports/figures/confusion_matrix.png'),
                      labels=parties
                      )
 #3. Print results
print(confusion_matrix(test_y, predictions))
print(classification_report(test_y, predictions))
print("Accurancy in test: {:.2f} %".format(100*(accuracy_score(test_y, predictions))))
