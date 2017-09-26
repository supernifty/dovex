# -*- coding: utf-8 -*-

import collections
import csv
import math
import os
import uuid

import flask

#import plotly.graph_objs as go
#import pandas as pd

app = flask.Flask(__name__, template_folder='templates')
app.config.from_pyfile('config.py')
app.secret_key = 'ducks in space'

@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in flask.request.files:
            flask.abort(400, 'No file part')
            return flask.redirect(flask.request.url)
        file = flask.request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flask.abort(400, 'No selected file')
            return flask.redirect(flask.request.url)
        if file:
            filename = str(uuid.uuid4())
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            # process file
            return flask.redirect(flask.url_for('process', filename=filename))
    return flask.render_template('main.html')

@app.route('/process/<filename>')
def process(filename):
    df = {}
    header = []
    row_count = 0
    try:
      with open(os.path.join(app.config['UPLOAD_FOLDER'], filename)) as data_fh:
        for row_count, row in enumerate(csv.reader(data_fh)):
          if row[0].startswith('#'):
            if row_count == 0:
              header = row
              for column in header:
                df[column] = {'missing': 0, 'unique': collections.defaultdict(int)}
          else:
            for idx, value in enumerate(row):
              colname = header[idx]
              df[colname]['unique'][value] += 1
              if value == '':
                df[colname]['missing'] += 1
      columns = []
      for column in header:
        missing = df[column]['missing']
        information = -sum([ df[column]['unique'][value] / row_count * math.log(df[column]['unique'][value] / row_count, 2) for value in df[column]['unique']])
        columns.append({'name': column, 'missing': missing * 100 / row_count, 'unique_count': len(df[column]['unique']), 'information': information})

      # TODO
      # - missing data by sample
      # - histograms of each column
      # - correlation between columns
      # - clustering

      return flask.render_template('process.html', row_count=row_count, column_count=len(columns), columns=columns)
    except FileNotFoundError:
      flask.abort(404)

if __name__ == '__main__':
    app.run()
