#!/usr/bin/env python
'''
  statistical learning and analysis methods
'''

import collections
import csv
import json
import math
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
    'mds': 1000,
    'tsne': 10000
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
    return project(data_fh, config, projector, has_features=False, max_rows=MAX_ROWS['tsne'])

def is_empty(x):
  return x in ('', 'NA')

def prep_correlation(data_fh, config):
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
        if row[0].startswith('#'): # comment
            continue
        for idx, cell in enumerate(row): # each col
            #if exclude_missing and cell == '':
            #  continue
            if y_exclude is not None and idx not in y_exclude: # row index to exclude
                colname = meta['header'][idx]
                data[colname].append(cell)
                if idx in categorical_cols: # count categorical
                  if colname not in counts:
                    counts[colname] = {}
                  if cell not in counts[colname]:
                    counts[colname][cell] = 0
                  counts[colname][cell] += 1
    return data, counts, meta, categorical_cols

def correlation(data_fh, config, with_detail=False):
    '''
        calculate correlation as a p-value of each feature
    '''
    data, counts, meta, categorical_cols = prep_correlation(data_fh, config)

    # calculate p-values
    xs = []
    cs = []
    ds = []
    ts = []
    zs = []
    categorical_col_names = set([meta['header'][i] for i in categorical_cols])
    for x in sorted(data.keys()): # x is colname
        xs.append(x)
        current = []
        current_cs = []
        current_ts = []
        current_ds = [] # details
        for y in sorted(data.keys()): # y is colname
            if x == y:
              current.append(0)
              current_cs.append(0)
              current_ts.append('N/A')
              current_ds.append('N/A')
            else:
              try:
                if x in categorical_col_names and y in categorical_col_names: # chi-square
                    observed = collections.defaultdict(int)
                    expected_x = collections.defaultdict(int)
                    expected_y = collections.defaultdict(int)
                    for idx, _ in enumerate(data[x]): # values from x column
                      if is_empty(data[x][idx]) or is_empty(data[y][idx]): # skip if either are empty
                        continue
                      key = (data[x][idx], data[y][idx])
                      observed[key] += 1
                      expected_x[data[x][idx]] += 1
                      expected_y[data[y][idx]] += 1

                    # unobserved combinations
                    ks = list(observed.keys())
                    for k in ks:
                      xkey = k[0]
                      for l in ks:
                        ykey = l[1]
                        key = (xkey, ykey)
                        if key not in observed:
                          observed[key] = 0
  
                    total_observed = sum([observed[key] for key in observed])
                    current_cs.append(total_observed)
                    current_ts.append('Chi-square')
                    if total_observed > 0:
                      dof = (len(expected_x) - 1) * (len(expected_y) - 1) # correct dof
                      ddof = len(observed) - 1 - dof
                      pvalue = scipy.stats.chisquare([observed[key] for key in sorted(observed.keys())], [expected_x[key[0]] * expected_y[key[1]] / total_observed for key in sorted(observed.keys())], ddof=ddof)[1]
                      current_ds.append(' '.join(['{}/{}={}'.format(key[0], key[1], observed[key]) for key in sorted(observed.keys())]))
                    else:
                      pvalue = 1
                      current_ds.append('N/A')
                elif x not in categorical_col_names and y not in categorical_col_names: # both numeric: pearson correlation
                    v1s = []
                    v2s = []
                    for idx, _ in enumerate(data[x]):
                      if is_empty(data[x][idx]) or is_empty(data[y][idx]): # skip if either are empty
                        continue
                      v1s.append(float(data[x][idx]))
                      v2s.append(float(data[y][idx]))
                    current_cs.append(len(v1s))
                    current_ts.append('Pearson correlation')
                    current_ds.append('N/A')
                    if len(v1s) > 1 and len(v2s) > 1:
                      pvalue = scipy.stats.pearsonr(v1s, v2s)[1]
                    else:
                      pvalue = 1
                else: # one categorical - anova
                    groups = collections.defaultdict(list)
                    for idx, _ in enumerate(data[x]):
                      if is_empty(data[x][idx]) or is_empty(data[y][idx]): # skip if either are empty
                        continue
                      if x in categorical_col_names:
                        groups[data[x][idx]].append(float(data[y][idx]))
                      else:
                        groups[data[y][idx]].append(float(data[x][idx]))
                    current_cs.append(sum([len(groups[g]) for g in groups]))
                    current_ts.append('ANOVA')
                    current_ds.append('N/A')
                    if len(groups) > 1:
                      pvalue = scipy.stats.f_oneway(*[groups[k] for k in groups])[1]
                      if math.isnan(pvalue): # if all values the same
                        pvalue = 1
                    else:
                      pvalue = 1
              except:
                pvalue = -1 # problem
                raise
              if math.isnan(pvalue):
                pvalue = -1
              current.append(pvalue)
        zs.append(current)
        cs.append(current_cs)
        ds.append(current_ds)
        ts.append(current_ts)
    # xs: [cov1 cov2 cov3]
    # zs: [[pv11 pv12 pv13], [pv21 pv22 pv23]...]]
    # cs: [[n11 n12 n13], [n21 n22 n23]...]]
    # ts: [[t11 t12 t13], [t21 t22 t23]...]]
    return {'xs': xs, 'zs': zs, 'cs': cs, 'ts': ts, 'ds': ds}

def correlation_subgroup(data_fh, config):
  '''
    for anova or chi-square do individual breakdowns
  '''
  data, counts, meta, categorical_cols = prep_correlation(data_fh, config)
  # calculate p-values
  result = []
  categorical_col_names = set([meta['header'][i] for i in categorical_cols])
  y_include_1 = config['x_exclude'] # straight out column name
  y_include_2 = config['y_predict'] # straight out column name
  added = set()
  for x in sorted(data.keys()): # x is colname
    for y in sorted(data.keys()): # y is colname
      if x == y:
        continue
      else:
        if x == y_include_1 and y == y_include_2 and x in categorical_col_names and y in categorical_col_names: # chi-square
          observed = collections.defaultdict(int)
          expected_x = collections.defaultdict(int)
          expected_y = collections.defaultdict(int)

          subgroups = set()
          # counts of each combination
          for idx, _ in enumerate(data[x]):
            if is_empty(data[x][idx]) or is_empty(data[y][idx]): # skip if either are empty
              continue
            key = (data[x][idx], data[y][idx])
            observed[key] += 1
            subgroups.add(data[x][idx]) # category of primary covariate
            expected_x[data[x][idx]] += 1
            expected_y[data[y][idx]] += 1

          # unobserved combinations
          ks = list(observed.keys())
          for k in ks: # all combinations seen
            xkey = k[0] # came from data[x][idx]
            for l in ks:
              ykey = l[1]
              key = (xkey, ykey)
              if key not in observed:
                #print('adding zero for {}'.format(key))
                observed[key] = 0

          for s1 in subgroups:
            for s2 in subgroups:
              if s1 == s2 or (s2, s1) in added:
                continue
              added.add((s1, s2))
              total_observed = sum([observed[key] for key in observed if key[0] == s1 or key[0] == s2])
              if total_observed > 0:
                pvalue = scipy.stats.chisquare([observed[key] for key in sorted(observed.keys()) if key[0] == s1 or key[0] == s2], [expected_x[key[0]] * expected_y[key[1]] / total_observed for key in sorted(observed.keys()) if key[0] == s1 or key[0] == s2])[1]
              else:
                pvalue = 1
              result.append((s1, s2, pvalue, total_observed, '-', '-', '-', '-', 'Chi-square'))

        elif x == y_include_1 and y == y_include_2 and (x in categorical_col_names or y in categorical_col_names): # t-tests
          groups = collections.defaultdict(list)
          subgroups = set()
          for idx, _ in enumerate(data[x]):
            if is_empty(data[x][idx]) or is_empty(data[y][idx]): # skip if either are empty
              continue
            if x in categorical_col_names:
              groups[data[x][idx]].append(float(data[y][idx]))
              subgroups.add(data[x][idx])
            else:
              groups[data[y][idx]].append(float(data[x][idx]))
              subgroups.add(data[y][idx])

          for s1 in subgroups:
            for s2 in subgroups:
              if s1 == s2 or (s2, s1) in added:
                continue
              added.add((s1, s2))
              pvalue = scipy.stats.ttest_ind(groups[s1], groups[s2])[1]
              if math.isnan(pvalue): # if all values the same
                pvalue = 1
              result.append((s1, s2, pvalue, len(groups[s1]) + len(groups[s2]), np.mean(groups[s1]), np.mean(groups[s2]), np.std(groups[s1], ddof=1), np.std(groups[s2], ddof=1), 't-test'))

 
        else: # not applicable chi-square
          pass 
  return {'result': result}

METHODS = {
    'logistic': logistic_regression,
    'svc': svc,
    'rf': random_forest,
    'linear': linear_regression,
    'svr': svr,
    'pca': pca,
    'mds': mds,
    'tsne': tsne,
    'correlation': correlation,
    'correlation_subgroup': correlation_subgroup
}
