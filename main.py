# -*- coding: utf-8 -*-

import os
import uuid

import flask

import plotly.graph_objs as go
import pandas as pd

app = flask.Flask(__name__, template_folder='templates')
app.config.from_pyfile('config.py')
app.secret_key = 'ducks in space'

@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in flask.request.files:
            flash('No file part')
            return flask.redirect(flask.request.url)
        file = flask.request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return flask.redirect(flask.request.url)
        if file:
            filename = str(uuid.uuid4())
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            # process file
            return flask.redirect(flask.url_for('process', filename=filename))
    return flask.render_template('main.html')

@app.route('/process/<filename>')
def process(filename):
    df = pd.read_csv(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    row_count, column_count = df.shape
    columns = []
    for column in df:
      missing = sum(df[column].isnull())
      columns.append({'name': column, 'type': df[column].dtype, 'missing': missing * 100 / row_count, 'unique_count': len(df[column].unique()), 'unique': ', '.join([str(x) for x in df[column].unique()[:5]])})
    return flask.render_template('process.html', row_count=row_count, column_count=column_count, columns=columns)

if __name__ == '__main__':
    app.run()
