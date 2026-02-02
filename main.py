# -*- coding: utf-8 -*-
'''
    main web interface
'''

import csv
import datetime
import io
import os
import re
import requests
import secrets
import urllib
import uuid
import argparse
import pandas as pd
import numpy as np
from pandas.core.dtypes.common import infer_dtype_from_object
import sqlite3

import flask
import flask_login

import ml
import proxy
import util

#import plotly.graph_objs as go
#import pandas as pd

NOAUTH='main'
DEFAULT_USER='dovex@supernifty.org'

MISSINGNESS=False

app = flask.Flask(__name__, template_folder='templates')
app.config.from_pyfile('config.py')
app.secret_key = 'ducks in space'
app.wsgi_app = proxy.ReverseProxied(app.wsgi_app)

login = flask_login.LoginManager(app)
login.login_view = NOAUTH

# ----- auth database -----
class User (flask_login.UserMixin):
  _email = None

  def __init__(self, email):
    self._email = email

  def is_authenticated(self):
    return True

  def is_active(self):
    return True

  def is_anonymous(self):
    return False

  def get_id(self):
    return self._email

def create_database():
  # save the metadata - create if not exist
  db = os.path.join(app.config['UPLOAD_FOLDER'], app.config['META'])
  con = sqlite3.connect(db)    
  cursor = con.cursor()
  cursor.execute('''
      CREATE TABLE IF NOT EXISTS users (email text primary key not null)''')
  cursor.execute('''
      CREATE TABLE IF NOT EXISTS dataset (
          filename text primary key not null,
          email text not null,
          title text not null,
          created text not null,
          size integer not null)''')
  con.commit()
  return con

#def users():
#  db = os.path.join(app.config['UPLOAD_FOLDER'], app.config['META'])
#  con = sqlite3.connect(db)
#  return con

def find_or_create_user(email):
  con = create_database()
  cursor = con.cursor()
  cursor.execute("select email from users where email = ?", (email,))
  existing = cursor.fetchone()
  if existing is None:
    # add new user
    cursor.execute('insert into users values (?)', (email,))
    con.commit()
    return User(email)
  else:
    return User(existing[0])

@login.user_loader
def load_user(user_id):
    cursor = create_database().cursor()
    cursor.execute("select email from users where email = ?", (user_id,))
    existing = cursor.fetchone()
    if existing is None:
      return None
    else:
      return User(existing[0])

@app.route('/', methods=['GET', 'POST'])
def main():
    '''
        saves the uploaded file and forwards to the processor
    '''
    if flask.request.method == 'POST':
        if not flask_login.current_user.is_authenticated:
            flask.abort(403)
        # check if the post request has the file part
        if 'file' not in flask.request.files:
            flask.abort(400, 'No file part')
            return flask.redirect(flask.request.url)
        data = flask.request.files['file']
        title = flask.request.form['title']
        # if user does not select file, browser also
        # submit a empty part without filename
        if data.filename == '':
            flask.abort(400, 'No selected file')
            return flask.redirect(flask.request.url)
        if data:
            filename = str(uuid.uuid4())
            write(data, filename, title=title)
            # process file
            return flask.redirect(flask.url_for('explore', filename=filename))
    return flask.render_template('main.html')

@app.route('/uploads', methods=['GET'])
@flask_login.login_required
def uploads():
  db = os.path.join(app.config['UPLOAD_FOLDER'], app.config['META'])
  con = sqlite3.connect(db)
  cursor = con.cursor()
  items = []
  for item in cursor.execute('select filename, title, created, size from dataset where email = ?', (flask_login.current_user._email,)):
    items.append({'filename': item[0], 'title': item[1], 'created': item[2], 'size': item[3]})
  return flask.render_template('uploads.html', items=items)

@app.route('/delete/<filename>', methods=['GET'])
@flask_login.login_required
def delete(filename):
  target_fn = os.path.join(app.config['UPLOAD_FOLDER'], filename) 
  if os.path.exists(target_fn):
    os.remove(target_fn)
    db = os.path.join(app.config['UPLOAD_FOLDER'], app.config['META'])
    con = sqlite3.connect(db)
    cursor = con.cursor()
    cursor.execute('delete from dataset where filename = ? and email = ?', (filename, flask_login.current_user._email))
    con.commit()
  return flask.redirect(flask.url_for('uploads'))

def write(data, filename, title=''):
    # save the data
    target_fn = os.path.join(app.config['UPLOAD_FOLDER'], filename) 
    data.save(target_fn)
    size = os.stat(target_fn).st_size

    con = create_database()
    cursor = con.cursor()
    # now insert
    cursor.execute('insert into dataset values (?, ?, ?, ?, ?)', (filename, flask_login.current_user._email, title, datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), size))
    con.commit()

def read(filename):
    '''
        opens specified filehandle
        currently only supports files
    '''
    if filename == 'meta.sqlite':
      raise FileNotFoundError

    #if filename.startswith('url='):
    #  return urllib.urlopen(filename[4:])
    #else:
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
                     for (column,coltype) in df.dtypes.items()]
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


@app.route('/config/<filename>')
def config(filename):
    '''
        store configuration options for this dataset
    '''
    # TODO
    flask.abort(404)
    

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

        # read datasest
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

        # metadata derived from the dataset
        meta['header'] = list(df.columns)
        meta['lines'] = len(df)
        meta['datatype'] = list(guess_datatypes(df, known_datatypes=datatype_row))

        # TODO get dataset configuration

        # TODO: if desired, set variables with missing values to categorical
        if MISSINGNESS:
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
        df_str[df.isnull()] = '' # TODO??? converting NA etc to ''?
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

##### auth #####
@app.route('/logout')
@flask_login.login_required
def logout():
  flask_login.logout_user()
  flask.flash('You have been logged out.')
  return flask.redirect(flask.url_for(NOAUTH))

@app.route('/authorize/<provider>')
def oauth2_authorize(provider):
    if 'AUTH' not in app.config or app.config['AUTH'] == 'none':
      user = find_or_create_user(DEFAULT_USER)
      # log the user in
      flask_login.login_user(user)
      return flask.redirect(flask.url_for('main')) # TODO redirect to target?

    if not flask_login.current_user.is_anonymous:
        return flask.redirect(flask.url_for(NOAUTH))

    provider_data = app.config['OAUTH2_PROVIDERS'].get(provider)
    if provider_data is None:
        flask.abort(404)

    # generate a random string for the state parameter
    flask.session['oauth2_state'] = secrets.token_urlsafe(16)

    # create a query string with all the OAuth2 parameters
    qs = urllib.parse.urlencode({
        'client_id': provider_data['client_id'],
        'redirect_uri': flask.url_for('oauth2_callback', provider=provider, _external=True),
        'response_type': 'code',
        'scope': ' '.join(provider_data['scopes']),
        'state': flask.session['oauth2_state'],
    })

    # redirect the user to the OAuth2 provider authorization URL
    return flask.redirect(provider_data['authorize_url'] + '?' + qs)

@app.route('/callback/<provider>')
def oauth2_callback(provider):
    if not flask_login.current_user.is_anonymous:
        return flask.redirect(flask.url_for(NOAUTH))

    provider_data = app.config['OAUTH2_PROVIDERS'].get(provider)
    if provider_data is None:
        flask.abort(404)

    # if there was an authentication error, flash the error messages and exit
    if 'error' in flask.request.args:
        for k, v in flask.request.args.items():
            if k.startswith('error'):
                flask.flash(f'{k}: {v}')
        return flask.redirect(flask.url_for(NOAUTH))

    # make sure that the state parameter matches the one we created in the
    # authorization request
    if flask.request.args['state'] != flask.session.get('oauth2_state'):
        flask.abort(401)

    # make sure that the authorization code is present
    if 'code' not in flask.request.args:
        flask.abort(401)

    # exchange the authorization code for an access token
    response = requests.post(provider_data['token_url'], data={
        'client_id': provider_data['client_id'],
        'client_secret': provider_data['client_secret'],
        'code': flask.request.args['code'],
        'grant_type': 'authorization_code',
        'redirect_uri': flask.url_for('oauth2_callback', provider=provider, _external=True),
    }, headers={'Accept': 'application/json'})
    if response.status_code != 200:
        flask.abort(401)
    oauth2_token = response.json().get('access_token')
    if not oauth2_token:
        flask.abort(401)

    # use the access token to get the user's email address
    response = requests.get(provider_data['userinfo']['url'], headers={
        'Authorization': 'Bearer ' + oauth2_token,
        'Accept': 'application/json',
    })
    if response.status_code != 200:
        flask.abort(401)
    email = provider_data['userinfo']['email'](response.json())

    # find or create the user in the database
    user = find_or_create_user(email)

    # log the user in
    flask_login.login_user(user)
    return flask.redirect(flask.url_for('main')) # TODO redirect to target?

if __name__ == '__main__':
    app.run(host='0.0.0.0')
    #app.run()
