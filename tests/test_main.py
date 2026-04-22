"""
Tests for main.py Flask application routes and data handling.
"""

import flask

import main


def test_json_data_iris():
    main.app.config['TESTING'] = True
    client = main.app.test_client()
    with main.app.app_context():
        response = client.get('/data/iris.data')
        result = flask.json.loads(response.data)
        assert len(result['data']) == 150
        assert result['meta']['lines'] == 150


def test_uploads_template_renders_one_row_per_dataset():
    items = [
        {'filename': 'first.csv', 'title': 'First', 'created': '2026-04-22 10:00:00', 'size': 123},
        {'filename': 'second.csv', 'title': 'Second', 'created': '2026-04-22 11:00:00', 'size': 456},
    ]

    with main.app.test_request_context('/uploads'):
        rendered = flask.render_template('uploads.html', items=items)

    # One header row plus one row for each dataset
    assert rendered.count('<tr>') == 1 + len(items)
    assert '>First<' in rendered
    assert '>Second<' in rendered


def test_load_app_config_applies_optional_local_override(tmp_path):
    default_config = tmp_path / 'config.py'
    local_config = tmp_path / 'config.local.py'
    default_config.write_text("DEBUG = False\nUPLOAD_FOLDER = './uploads'\n")
    local_config.write_text("DEBUG = True\nAUTH = 'none'\n")

    app = flask.Flask(__name__, root_path=str(tmp_path))
    main.load_app_config(app, default_config='config.py', local_config='config.local.py')

    assert app.config['DEBUG'] is True
    assert app.config['UPLOAD_FOLDER'] == './uploads'
    assert app.config['AUTH'] == 'none'


def test_load_app_config_ignores_missing_local_override(tmp_path):
    default_config = tmp_path / 'config.py'
    default_config.write_text("DEBUG = False\nUPLOAD_FOLDER = './uploads'\n")

    app = flask.Flask(__name__, root_path=str(tmp_path))
    main.load_app_config(app, default_config='config.py', local_config='config.local.py')

    assert app.config['DEBUG'] is False
    assert app.config['UPLOAD_FOLDER'] == './uploads'
    assert 'AUTH' not in app.config
