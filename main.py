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
    '''
        saves the uploaded file and forwards to the processor
    '''
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
            return flask.redirect(flask.url_for('explore', filename=filename))
    return flask.render_template('main.html')

@app.route('/data/<filename>')
def data(filename):
    '''
        provide details as json
    '''
    try:
        meta = {}
        data = []
        lines = 0

        with open(os.path.join(app.config['UPLOAD_FOLDER'], filename)) as data_fh:
            for lines, row in enumerate(csv.reader(data_fh)):
                if lines == 0:
                    meta['header'] = row
                    continue
                if row[0].startswith('#'):
                    continue
                data.append(row)

        meta['lines'] = lines
        return flask.jsonify(meta=meta, data=data)
    except FileNotFoundError:
        flask.abort(404)

@app.route('/explore/<filename>')
def explore(filename):
    '''
        process data client side
    '''
    try:
        return flask.render_template('explore.html', filename=filename)
    except FileNotFoundError:
        flask.abort(404)

if __name__ == '__main__':
    app.run()
