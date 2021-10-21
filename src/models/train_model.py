"""Train model functionality"""
from os import path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

def train_model():

    def testModel(train_X, train_y, word_model):
        pipeline = Pipeline([
            ('prepro', word_model),
            ('model', MultinomialNB())])
        params = {'prepro__min_df':[1, 2, 3, 5, 6],
                  'prepro__max_df':[0.75, 0.85, 0.95, 0.98],
                  'prepro__ngram_range':[(1, 1), (1, 2), (1, 3)]}
        grid = GridSearchCV(pipeline, param_grid=params, cv=5)
        fit_grid = grid.fit(train_X, train_y)
        print("Mean crossvalidation {}-accurancy on validation data {: .2f} %".format(
            word_model, 100*fit_grid.best_score_))
        print("Best params {} %".format(fit_grid.best_estimator_))
        return grid, fit_grid

    def word_to_num_info(grid, bag=False):
        vece = grid.best_estimator_.named_steps['prepro']
        train_X_tr = vece.transform(train_X)
        max_val = train_X_tr.max(axis=0).toarray().ravel()
        sorted_max_val = max_val.argsort()
        feature_names_s = np.array(vece.get_feature_names())
        if bag == True:
            print("Amount of features in bag: {}".format(len(feature_names_s)))
            print("\nFirst 50 features: {}".format(feature_names_s[0:50]))
            print("\nLast 20 features: {}".format(feature_names_s[-20:-1]))
        if bag == False:
            sorted_by_idf = np.argsort(vece.idf_)
            print('Features with lowest if (not relevant): \n {}'.format(
                feature_names_s[sorted_by_idf [:40]]))
            print('\nFeatures with lowest tfif (least impact): \n {}'.format(
                feature_names_s[sorted_max_val[:10]]))
            print('\nFeatures with highest tfif(highest impact): \n {}'.format(
                feature_names_s[sorted_max_val[:-10]]))

    #1. Import data
    base_path='/home/chpatola/Desktop/Skola/Python/cookie_nlp/'
    yle_data = pd.read_csv(path.join(base_path,'data/processed/processed_data.csv'),
                           sep=',',
                           encoding="ISO-8859-1"
                           )

    #2 Divide in X and y
    X = yle_data.work_for
    y = yle_data.party

    #3. Divide in train and test
    train_X, test_X, train_y, test_y = train_test_split(X,
                                                        y,
                                                        test_size=0.2,
                                                        random_state=2,
                                                        stratify=y
                                                        )

    #4. Bag of Words for train_X & evaluate performance
    bag, bag_results = testModel(train_X,
                                train_y,
                                CountVectorizer()
                                )
    word_to_num_info(bag_results, bag=True)

    #5. tf-idf for train X & evaluate performance
    tf_idf, tf_idf_results = testModel(train_X,
                                       train_y,
                                       TfidfVectorizer()
                                       )
    word_to_num_info(tf_idf)
    return test_X, test_y, bag, tf_idf
    