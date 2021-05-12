#!/usr/bin/env python
'''
  statistical learning and analysis methods
'''

import collections
import csv
import json
import sys
import util

import numpy as np

import scipy
import scipy.stats

import sklearn.ensemble
import sklearn.decomposition
import sklearn.linear_model
import sklearn.manifold
import sklearn.preprocessing
import sklearn.svm
import sklearn.model_selection

MAX_DISTINCT = 100
MAX_CELLS = 1e6
MAX_ROWS = {
    'mds': 1000
}

CLASS_WEIGHT_MAP = {
  'unadjusted': None,
  'balanced': 'balanced'
}

def preprocess(data_fh, config):
    '''
        puts the input data into X and y in an appropriate format for analysis
        one hot encoding, normalization
    '''
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

    delimiter = util.choose_delimiter(data_fh)
    impute_numeric = 0
    for lines, row in enumerate(csv.reader(data_fh, delimiter=delimiter)):
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
                if idx in categorical_cols: # categorical
                    chunk = [0] * distinct[idx]
                    if cell in seen[idx]:
                        chunk[seen[idx][cell]] = 1
                    else:
                        chunk[seen_count[idx]] = 1
                        seen[idx][cell] = seen_count[idx]
                        seen_count[idx] += 1
                    x.extend(chunk)
                elif cell == '': # handle missing float
                    #return {'error': 'Cannot train with missing data in numeric column {} on line {}.'.format(idx + 1, lines + 1)}
                    x.append(np.nan)
                    impute_numeric += 1
                else:
                    try:
                        x.append(float(cell))
                    except ValueError:
                        x.append(np.nan)
                        impute_numeric += 1

        if missing < x_exclude: # sufficient number of populated columns
            if y_predict is not None:
                if y_predict in categorical_cols:
                    y.append(row[y_predict])
                    y_labels.add(row[y_predict])
                else:
                    y.append(float(row[y_predict]))
            X.append(x)

    # check limits
    if len(X) == 0:
        return {'error': 'No rows to analyze. Is there too much missing data?'}

    if len(X) * len(X[0]) > MAX_CELLS:
        return {'error': 'This dataset is too large ({} cells vs {} max). Reduce the number of rows or columns.'.format(len(X) * len(X[0]), MAX_CELLS)}

    # impute
    notes = []
    if impute_numeric > 0:
        imputer = sklearn.preprocessing.Imputer(missing_values=np.nan, strategy='mean', copy=False)
        X = imputer.fit_transform(X)
        notes.append('{} missing or erroneous numeric value(s) have been imputed with the mean of that column'.format(impute_numeric))

    # scale
    if 'scale' in config:
        X = sklearn.preprocessing.scale(X)

    return {'X': X, 'y_labels': y_labels, 'y': y, 'y_predict': y_predict, 'y_exclude': y_exclude, 'categorical_cols': categorical_cols, 'distinct': distinct, 'notes': notes}

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
        'training_score': learner.score(pre['X'], pre['y']),
        'notes': pre['notes']
    }

    if pre['y_predict'] in pre['categorical_cols']: # use accuracy
        scores = sklearn.model_selection.cross_val_score(learner, pre['X'], pre['y'], cv=3, scoring='accuracy')
        cv_predictions = sklearn.model_selection.cross_val_predict(learner, pre['X'], pre['y'], cv=3)
        confusion = sklearn.metrics.confusion_matrix(pre['y'], cv_predictions, labels=list(pre['y_labels']))
        result['confusion'] = confusion.tolist()
        result['y_labels'] = [' {}'.format(y) for y in list(pre['y_labels'])] # plotly acts weird for purely numeric labels
    else: # use MSE
        scores = sklearn.model_selection.cross_val_score(learner, pre['X'], pre['y'], cv=3, scoring='r2')

    # feature importance
    if learner_features is not None:
        result['features'] = map_to_original_features(learner_features(learner), pre['y_exclude'], pre['y_predict'], pre['distinct'], pre['categorical_cols'])

    result['cross_validation_score'] = scores.mean()

    return result

def project(data_fh, config, projector, has_features=True, max_rows=None):
    '''
        reduce dimensionality
    '''
    pre = preprocess(data_fh, config)
    if 'error' in pre:
        return pre

    if max_rows is not None and len(pre['X']) > max_rows:
        return {'error': 'Too many rows for this method: {} > {}'.format(len(pre['X']), max_rows)}

    projection = projector.fit_transform(pre['X'])
    if has_features:
        result = {
            'projection': projection.tolist(),
            'features': map_to_original_features(projection_features(projector), pre['y_exclude'], pre['y_predict'], pre['distinct'], pre['categorical_cols']),
            'features_2': map_to_original_features(projection_features(projector, component=1), pre['y_exclude'], pre['y_predict'], pre['distinct'], pre['categorical_cols']),
            'notes': pre['notes']
        }
    else:
        result = {
            'projection': projection.tolist(),
            'notes': pre['notes']
        }
    return result

def projection_features(projector, component=0):
    '''
        returns weighting of projection input features on component 0
    '''
    #return [x*x for x in projector.components_[component]]
    return [abs(x) for x in projector.components_[component]]

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
    #learner = sklearn.linear_model.LogisticRegression(C=1e5)
    learner = sklearn.linear_model.LogisticRegression(solver='lbfgs', multi_class='multinomial', class_weight=CLASS_WEIGHT_MAP[config.get('class_weight', 'unadjusted')])
    return evaluate(data_fh, config, learner, logistic_regression_features)

def svc(data_fh, config):
    '''
        perform evaluation using svm
    '''
    learner = sklearn.svm.LinearSVC(class_weight=CLASS_WEIGHT_MAP[config.get('class_weight', 'unadjusted')])
    return evaluate(data_fh, config, learner, svc_features)

def random_forest(data_fh, config):
    '''
        perform evaluation using rf
    '''
    learner = sklearn.ensemble.RandomForestClassifier(class_weight=CLASS_WEIGHT_MAP[config.get('class_weight', 'unadjusted')])
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
    return project(data_fh, config, projector, has_features=True)

def mds(data_fh, config):
    '''
        cluster data using mds
    '''
    projector = sklearn.manifold.MDS(n_components=2, max_iter=100, verbose=1)
    return project(data_fh, config, projector, has_features=False, max_rows=MAX_ROWS['mds'])

def tsne(data_fh, config):
    '''
        cluster data using tsne
    '''
    projector = sklearn.manifold.TSNE(n_components=2, verbose=1, perplexity=int(config['perplexity']), n_iter=300)
    return project(data_fh, config, projector, has_features=False, max_rows=MAX_ROWS['mds'])

def correlation(data_fh, config):
    '''
        calculate correlation as a p-value of each feature
    '''
    y_exclude = set([int(x) for x in json.loads(config['y_exclude'])])
    categorical_cols = set([i for i, x in enumerate(json.loads(config['datatype'])) if x == 'categorical'])
    delimiter = util.choose_delimiter(data_fh)
    meta = {}
    data = collections.defaultdict(list)
    counts = {}
    missing = 0
    # read in all the data
    for lines, row in enumerate(csv.reader(data_fh, delimiter=delimiter)): # each row
        if lines == 0:
            meta['header'] = row
            continue
        if len(row) == 0: # skip empty lines
            continue
        if row[0].startswith('#'):
            continue
        for idx, cell in enumerate(row): # each col
            if idx not in y_exclude:
                colname = meta['header'][idx]
                data[colname].append(cell)
                if idx in categorical_cols: # count categorical
                  if colname not in counts:
                    counts[colname] = {}
                  if cell not in counts[colname]:
                    counts[colname][cell] = 0
                  counts[colname][cell] += 1

    # calculate p-values
    xs = []
    zs = []
    categorical_col_names = set([meta['header'][i] for i in categorical_cols])
    for x in sorted(data.keys()): # x is colname
        xs.append(x)
        current = []
        for y in sorted(data.keys()): # y is colname
            if x == y:
              current.append(0)
            else:
              if x in categorical_col_names and y in categorical_col_names: # chi-square
                  observed = collections.defaultdict(int)
                  expected = {}
                  for idx, _ in enumerate(data[x]):
                    if data[x][idx] == '' or data[y][idx] == '':
                      continue
                    key = (data[x][idx], data[y][idx])
                    observed[key] += 1

                  pvalue = scipy.stats.chisquare([observed[key] for key in sorted(observed.keys())], [counts[x][key[0]] * counts[y][key[1]] / len(data) for key in sorted(observed.keys())])[1]
              elif x not in categorical_col_names and y not in categorical_col_names: # both numeric: pearson correlation
                  v1s = []
                  v2s = []
                  for idx, _ in enumerate(data[x]):
                    if data[x][idx] == '' or data[y][idx] == '':
                      continue
                    v1s.append(float(data[x][idx]))
                    v2s.append(float(data[y][idx]))
                  pvalue = scipy.stats.pearsonr(v1s, v2s)[1]
              else: # one categorical - anova
                  groups = collections.defaultdict(list)
                  for idx, _ in enumerate(data[x]):
                    if data[x][idx] == '' or data[y][idx] == '':
                      continue
                    if x in categorical_col_names:
                      groups[data[x][idx]].append(float(data[y][idx]))
                    else:
                      groups[data[y][idx]].append(float(data[x][idx]))
                  pvalue = scipy.stats.f_oneway(*[groups[k] for k in groups])[1]

              current.append(pvalue) # todo
        zs.append(current)
    return {'xs': xs, 'zs': zs}

METHODS = {
    'logistic': logistic_regression,
    'svc': svc,
    'rf': random_forest,
    'linear': linear_regression,
    'svr': svr,
    'pca': pca,
    'mds': mds,
    'tsne': tsne,
    'correlation': correlation
}
