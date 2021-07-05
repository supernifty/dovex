# -*- coding: utf-8 -*-
'''
    main web interface
'''

import csv
import io
import os
import re
import urllib
import uuid
import argparse
import pandas as pd
import numpy as np
from pandas.core.dtypes.common import infer_dtype_from_object

import boto3
import flask

import ml
import proxy
import util

#import plotly.graph_objs as go
#import pandas as pd

# arguments are currently used only to make missingness optional
parser = argparse.ArgumentParser()
parser.add_argument('--missingness', dest='missingness', action='store_true',
                    help='create extra variables corresponding to missingness of variables')
args = parser.parse_args()

app = flask.Flask(__name__, template_folder='templates')
app.config.from_pyfile('config.local.py')
app.secret_key = 'ducks in space'
app.wsgi_app = proxy.ReverseProxied(app.wsgi_app)

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
        data = flask.request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if data.filename == '':
            flask.abort(400, 'No selected file')
            return flask.redirect(flask.request.url)
        if data:
            filename = str(uuid.uuid4())
            write(data, filename)
            # process file
            return flask.redirect(flask.url_for('explore', filename=filename))
    return flask.render_template('main.html')


def write(data, filename):
    if app.config['AWS']:
      s3 = boto3.client('s3', aws_access_key_id=app.config['AWS_ACCESS'], aws_secret_access_key=app.config['AWS_SECRET'])
      s3.upload_fileobj(data, app.config['AWS_BUCKET'], filename)
    else:
      data.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

def read(filename):
    '''
        opens specified filehandle
        currently only supports files
    '''
    if app.config['AWS']:
      if filename.startswith('url='):
        filename = filename[4:]
      s3 = boto3.client('s3', aws_access_key_id=app.config['AWS_ACCESS'], aws_secret_access_key=app.config['AWS_SECRET'])
      s3_object = s3.get_object(Bucket=app.config['AWS_BUCKET'], Key=filename)
      return io.StringIO(s3_object['Body'].read().decode('utf-8'))
    else:
      if filename.startswith('url='):
        return urllib.urlopen(filename[4:])
      else:
        if re.match(r'^[\w.-]+$', filename) is None:
          raise FileNotFoundError
        return open(os.path.join(app.config['UPLOAD_FOLDER'], filename))

def matches_dtypes(df, dtypes):
    """
    Return Series that is True where dtype of column matches given dtypes.
    Returns a Series with index matching df.columns.
    dtypes must be an iterable of type specifications.
    Copies match logic from pandas.DataFrame.select_dtypes();
    use same type specifications:
    * To select all *numeric* types use the numpy dtype ``numpy.number``
    * To select strings you must use the ``object`` dtype, but note that
      this will return *all* object dtype columns
    * See the `numpy dtype hierarchy
      <http://docs.scipy.org/doc/numpy/reference/arrays.scalars.html>`__
    * To select datetimes, use np.datetime64, 'datetime' or 'datetime64'
    * To select timedeltas, use np.timedelta64, 'timedelta' or
      'timedelta64'
    * To select Pandas categorical dtypes, use 'category'
    * To select Pandas datetimetz dtypes, use 'datetimetz' (new in 0.20.0),
      or a 'datetime64[ns, tz]' string
    """
    dtypes = list(map(infer_dtype_from_object, dtypes))
    boolean_list = [any([issubclass(coltype.type,t) for t in dtypes])
                     for (column,coltype) in df.dtypes.iteritems()]
    return pd.Series(boolean_list, index=df.columns)

# Guess unknown datatypes
# For now, if it's not numeric, it's categorical
# For now, just use pandas' guess at dtype
def guess_datatypes(df, known_datatypes=None):
    """
    Where known_datatypes is '', fill in with guessed datatypes.
    Return Series of resulting datatypes.
    Will be identical to known_datatypes if all datatypes were specified.
    """
    ALLOWED_VALUES = {'categorical','numeric',''}
    if known_datatypes is None:
        known_datatypes = ['']*df.shape[1]
    if len(set(known_datatypes) - ALLOWED_VALUES) > 0:
        raise ValueError("Unrecognised datatypes: {}".format(set(known_datatypes) - ALLOWED_VALUES))

    datatypes = pd.Series(known_datatypes, index=df.columns)
    unknown = [t=='' for t in known_datatypes]
    looks_numeric = matches_dtypes(df, [np.number])
    # for now either numeric or categorical
    looks_categorical = ~looks_numeric
    datatypes[unknown & looks_numeric] = 'numeric'
    datatypes[unknown & looks_categorical] = 'categorical'
    return datatypes


@app.route('/data/<filename>')
def json_data(filename):
    '''
        Read in the data file from disk, parse, provide data as json.
        Calculate datatypes where not provided.
        Currently allowed datatypes are: numeric, categorical.
        Optionally, based on args.missingness,
        add variables to represent missingness of original variables.
    '''
    try:
        meta = {}
        data = []
        lines = 0
        datatype_row = None

        with read(filename) as data_fh:
            df = pd.read_csv(data_fh, header=0, sep=util.choose_delimiter(data_fh))

        if str(df.iloc[0,0])[0]=='#':
            datatype_row = df.iloc[0,:]
            datatype_row[0] = datatype_row[0][1:]
            df = df.iloc[1:,:]
        else:
            datatype_row = None

        meta['header'] = list(df.columns)
        meta['lines'] = len(df)

        meta['datatype'] = list(guess_datatypes(df, known_datatypes=datatype_row))
        # TODO: if desired, set variables with missing values to categorical

        if args.missingness:
            # Create a missingness variable for every variable that has missing data.
            new_fields = []
            for field in meta['header']:
                if df[field].isnull().sum() > 0:
                    newfield = "missing_"+field
                    print("{} has missing values, creating {}".format(field,newfield))
                    df[newfield] = df[field].isnull()
                    new_fields.append(newfield)
            meta['header'] += new_fields
            meta['datatype'] += ['categorical']*len(new_fields)

        df_str = df[meta['header']].astype(str)
        df_str[df.isnull()] = ''
        data = [list(record) for record in df_str.to_records(index=False)]

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
            result = ml.METHODS[method](read(filename), flask.request.form)
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
    app.run(host='0.0.0.0')
    #app.run()
