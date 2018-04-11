# -*- coding: utf-8 -*-
'''
    main web interface
'''

import csv
import os
import re
import urllib
import uuid

import flask

import ml
import util

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


def get_fh(filename):
    '''
        opens specified filehandle
        currently only supports files
    '''
    if filename.startswith('url='):
        return urllib.urlopen(filename[4:])
    else:
        if re.match(r'^[\w.-]+$', filename) is None:
            raise FileNotFoundError
        return open(os.path.join(app.config['UPLOAD_FOLDER'], filename))

@app.route('/data/<filename>')
def json_data(filename):
    '''
        provide data as json
    '''
    try:
        meta = {}
        data = []
        lines = 0

        with get_fh(filename) as data_fh:
            delimiter = util.choose_delimiter(data_fh)
            for lines, row in enumerate(csv.reader(data_fh, delimiter=delimiter)):
                if lines == 0:
                    meta['header'] = row
                    continue
                if len(row) == 0: # skip empty lines
                    continue
                if row[0].startswith('#'):
                    if lines == 1:
                        meta['datatype'] = row
                        meta['datatype'][0] = row[0][1:] # remove leading #
                    continue
                data.append(row)

        meta['lines'] = lines
        return flask.jsonify(meta=meta, data=data)
    except FileNotFoundError:
        flask.abort(404)

@app.route('/explore/<filename>')
def explore(filename):
    '''
        client side explorer
    '''
    try:
        return flask.render_template('explore.html', filename=filename)
    except FileNotFoundError:
        flask.abort(404)

@app.route('/process/<filename>', methods=['POST'])
def process(filename):
    '''
        server side analysis
    '''
    try:
        method = flask.request.form['method']
        if method in ml.METHODS:
            result = ml.METHODS[method](get_fh(filename), flask.request.form)
            return flask.jsonify(result=result)
        else:
            flask.abort(404, 'method not found')
    except FileNotFoundError:
        flask.abort(404, 'data not found')

@app.route('/help')
def show_help():
    '''
        render help page
    '''
    return flask.render_template('help.html')

if __name__ == '__main__':
    #app.run(host='0.0.0.0')
    app.run()
