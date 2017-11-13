#!/usr/bin/env python
'''
  statistical learning and analysis methods
'''

import collections
import csv
import json

import sklearn.ensemble
import sklearn.linear_model
import sklearn.svm
import sklearn.model_selection

MAX_DISTINCT = 100
MAX_CELLS = 1e6

def evaluate(data_fh, config, learner):
    # x_exclude, y_predict, y_exclude, scale?
    y_exclude = set([int(x) for x in json.loads(config['y_exclude'])])
    x_exclude = set([int(x) for x in json.loads(config['x_exclude'])])
    y_predict = int(config['y_predict'])
    categorical_cols = set([i for i, x in enumerate(json.loads(config['datatype'])) if x == 'categorical'])
    distinct = { int(k): v for k, v in json.loads(config['distinct']).items() }

    # exclude columns with too many distincts
    for i in distinct.keys():
      if distinct[i] > MAX_DISTINCT:
        y_exclude.add(i)

    X = []
    y = []
    y_labels = set()
    meta = {}

    seen = collections.defaultdict(dict)
    seen_count = collections.defaultdict(int)
    
    for lines, row in enumerate(csv.reader(data_fh)):
      if lines in x_exclude:
        continue
      if lines == 0:
        meta['header'] = row
        continue
      if len(row) == 0: # skip empty lines
        continue
      if row[0].startswith('#'):
        continue
 
      # one hot and exclusions
      x = []
      for idx, cell in enumerate(row): # each col
        if idx not in y_exclude and idx != y_predict:
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
            return { 'error': 'Cannot train with missing data in numeric column {} on line {}.'.format(idx + 1, lines + 1) }
            break
          else:
            x.append(float(cell))

      if y_predict in categorical_cols:
        y.append(row[y_predict])
        y_labels.add(row[y_predict])
      else:
        y.append(float(row[y_predict]))

      X.append(x)

    # check limits
    if len(X) * len(X[0]) > MAX_CELLS:
        return { 'error': 'This dataset is too large ({} cells vs {} max). Reduce the number of rows or columns.'.format(len(X) * len(X[0]), MAX_CELLS) }

    # scale
    if 'scale' in config:
      X = sklearn.preprocessing.scale(X)

    learner.fit(X, y)
    predictions = learner.predict(X)

    result = { 
      'predictions': predictions.tolist(),
      'training_score': learner.score(X, y)
    }

    if y_predict in categorical_cols: # use accuracy
      scores = sklearn.model_selection.cross_val_score(learner, X, y, cv=5, scoring='accuracy')
      result['confusion'] = sklearn.metrics.confusion_matrix(y, predictions, labels=list(y_labels)).tolist()
      result['y_labels'] = list(y_labels)
    else: # use MSE
      scores = sklearn.model_selection.cross_val_score(learner, X, y, cv=5, scoring='r2')

    # feature importance

    result['cross_validation_score'] = scores.mean()
 
    return result

def logistic_regression(data_fh, config):
    learner = sklearn.linear_model.LogisticRegression(C=1e5)
    return evaluate(data_fh, config, learner)

def svc(data_fh, config):
    learner = sklearn.svm.LinearSVC()
    return evaluate(data_fh, config, learner)

def random_forest(data_fh, config):
    learner = sklearn.ensemble.RandomForestClassifier()
    return evaluate(data_fh, config, learner)


def linear_regression(data_fh, config):
    learner = sklearn.linear_model.LinearRegression()
    return evaluate(data_fh, config, learner)

def svr(data_fh, config):
    learner = sklearn.svm.LinearSVR()
    return evaluate(data_fh, config, learner)

METHODS = {
  'logistic': logistic_regression,
  'svc': svc,
  'rf': random_forest,
  'linear': linear_regression,
  'svr': svr
}

