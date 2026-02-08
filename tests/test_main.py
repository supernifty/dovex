
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
