#!/usr/bin/env python
'''
  statistical learning and analysis methods
'''

import collections
import csv
import json

import numpy as np

import sklearn.ensemble
import sklearn.decomposition
import sklearn.linear_model
import sklearn.svm
import sklearn.model_selection

MAX_DISTINCT = 100
MAX_CELLS = 1e6

def preprocess(data_fh, config):
    # x_exclude, y_predict, y_exclude, scale?
    y_exclude = set([int(x) for x in json.loads(config['y_exclude'])])
    x_exclude = int(config['x_exclude']) # max missing cols
    y_predict = None if config['y_predict'] == '' else int(config['y_predict'])
    categorical_cols = set([i for i, x in enumerate(json.loads(config['datatype'])) if x == 'categorical'])
    distinct = {int(k): v for k, v in json.loads(config['distinct']).items()}

    # exclude columns with too many distincts
    for i in distinct.keys():
        if i in categorical_cols and distinct[i] > MAX_DISTINCT:
            y_exclude.add(i)

    X = []
    y = []
    y_labels = set()
    meta = {}

    seen = collections.defaultdict(dict)
    seen_count = collections.defaultdict(int)
    
    for lines, row in enumerate(csv.reader(data_fh)):
        if lines == 0:
            meta['header'] = row
            continue
        if len(row) == 0: # skip empty lines
            continue
        if row[0].startswith('#'):
            continue

        # one hot and exclusions
        x = []
        missing = 0
        for idx, cell in enumerate(row): # each col
            if idx not in y_exclude and idx != y_predict:
                if cell == '': # missing
                    missing += 1
                    if missing >= x_exclude:
                        break # skip
                if idx in categorical_cols:
                    chunk = [0] * distinct[idx]
                    if cell in seen[idx]:
                        chunk[seen[idx][cell]] = 1
                    else:
                        chunk[seen_count[idx]] = 1
                        seen[idx][cell] = seen_count[idx]
                        seen_count[idx] += 1
                    x.extend(chunk)
                elif cell == '': # don't handle missing float
                    return {'error': 'Cannot train with missing data in numeric column {} on line {}.'.format(idx + 1, lines + 1)}
                else:
                    x.append(float(cell))

        if missing < x_exclude: # sufficient number of populated columns
            if y_predict is not None:
                if y_predict in categorical_cols:
                    y.append(row[y_predict])
                    y_labels.add(row[y_predict])
                else:
                    y.append(float(row[y_predict]))
            X.append(x)

    # check limits
    if len(X) * len(X[0]) > MAX_CELLS:
        return {'error': 'This dataset is too large ({} cells vs {} max). Reduce the number of rows or columns.'.format(len(X) * len(X[0]), MAX_CELLS)}

    # scale
    if 'scale' in config:
        X = sklearn.preprocessing.scale(X)

    return {'X': X, 'y_labels': y_labels, 'y': y, 'y_predict': y_predict, 'y_exclude': y_exclude, 'categorical_cols': categorical_cols, 'distinct': distinct}

def evaluate(data_fh, config, learner, learner_features=None):
    '''
        run prediction on over provided data and learner
    '''
    pre = preprocess(data_fh, config)
    if 'error' in pre:
        return pre

    learner.fit(pre['X'], pre['y'])
    predictions = learner.predict(pre['X'])

    result = {
        'predictions': predictions.tolist(),
        'training_score': learner.score(pre['X'], pre['y'])
    }

    if pre['y_predict'] in pre['categorical_cols']: # use accuracy
        scores = sklearn.model_selection.cross_val_score(learner, pre['X'], pre['y'], cv=5, scoring='accuracy')
        cv_predictions = sklearn.model_selection.cross_val_predict(learner, pre['X'], pre['y'], cv=5)
        confusion = sklearn.metrics.confusion_matrix(pre['y'], cv_predictions, labels=list(pre['y_labels']))
        result['confusion'] = confusion.tolist()
        result['y_labels'] = [' {}'.format(y) for y in list(pre['y_labels'])] # plotly acts weird for purely numeric labels
    else: # use MSE
        scores = sklearn.model_selection.cross_val_score(learner, pre['X'], pre['y'], cv=5, scoring='r2')

    # feature importance
    if learner_features is not None:
        result['features'] = map_to_original_features(learner_features(learner), pre['y_exclude'], pre['y_predict'], pre['distinct'], pre['categorical_cols'])

    result['cross_validation_score'] = scores.mean()

    return result

def project(data_fh, config, projector):
    '''
        reduce dimensionality
    '''
    pre = preprocess(data_fh, config)
    if 'error' in pre:
        return pre

    projection = projector.fit_transform(pre['X'])
    result = {
        'projection': projection.tolist(),
        'features': map_to_original_features(projection_features(projector), pre['y_exclude'], pre['y_predict'], pre['distinct'], pre['categorical_cols'])
    }
    return result

def projection_features(projector):
    return [x*x for x in projector.components_[0]]

def map_to_original_features(importances, y_exclude, y_predict, distinct, categorical_cols):
    '''
        convert one hot encoded back to correct index
    '''
    result = []
    current_col = 0
    importance = 0
    while importance < len(importances):
        if current_col in y_exclude or current_col == y_predict:
            result.append(0)
            current_col += 1
        else:
            if current_col in categorical_cols:
                number_of_columns = distinct[current_col]
                result.append(sum(importances[importance:importance + number_of_columns]))
                current_col += 1
                importance += number_of_columns
            else:
                result.append(importances[importance])
                importance += 1
                current_col += 1
    return result

# helpers
def logistic_regression_features(learner):
    '''
        importance of features for logistic regression
    '''
    return [x*x for x in learner.coef_[0]]

def svc_features(learner):
    '''
        importance of features for svm
    '''
    raw = learner.coef_
    result = np.sum(raw**2, axis=0)
    return result

def random_forest_features(learner):
    '''
        importance of features for rf
    '''
    return learner.feature_importances_

def linear_regression_features(learner):
    '''
        importance of features for linear regression
    '''
    return [x*x for x in learner.coef_]

def svr_features(learner):
    '''
        importance of features for svm
    '''
    return [x*x for x in learner.coef_]

# prediction algorithms

def logistic_regression(data_fh, config):
    '''
        perform evaluation using logistic regression
    '''
    learner = sklearn.linear_model.LogisticRegression(C=1e5)
    return evaluate(data_fh, config, learner, logistic_regression_features)

def svc(data_fh, config):
    '''
        perform evaluation using svm
    '''
    learner = sklearn.svm.LinearSVC()
    return evaluate(data_fh, config, learner, svc_features)

def random_forest(data_fh, config):
    '''
        perform evaluation using rf
    '''
    learner = sklearn.ensemble.RandomForestClassifier()
    return evaluate(data_fh, config, learner, random_forest_features)

def linear_regression(data_fh, config):
    '''
        perform evaluation using linear regression
    '''
    learner = sklearn.linear_model.LinearRegression()
    return evaluate(data_fh, config, learner, linear_regression_features)

def svr(data_fh, config):
    '''
        perform evaluation using svm
    '''
    learner = sklearn.svm.LinearSVR()
    return evaluate(data_fh, config, learner, svr_features)

# dimensionality reduction implementations

def pca(data_fh, config):
    '''
        cluster data using pca
    '''
    projector = sklearn.decomposition.PCA(n_components=2)
    return project(data_fh, config, projector)

METHODS = {
    'logistic': logistic_regression,
    'svc': svc,
    'rf': random_forest,
    'linear': linear_regression,
    'svr': svr,
    'pca': pca
}