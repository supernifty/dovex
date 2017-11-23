
var
  // example supervised learning algorithm
  ml = {
    // categorical prediction
    'logistic': function() { return server_side_predictor('logistic', 'categorical', 'Logistic Regression'); },
    'rf': function() { return server_side_predictor('rf', 'categorical', 'Random Forest'); },
    'svc': function() { return server_side_predictor('svc', 'categorical', 'Support Vector Classifier (SVC)'); },

    // numeric prediction
    'linear': function() { return server_side_predictor('linear', 'numeric', 'Linear Regression'); },
    'svr': function() { return server_side_predictor('svr', 'numeric', 'Support Vector Regression'); },

    // dimensionality reduction
    'pca': function() { return server_side_predictor('pca', 'reduce', 'PCA'); }
  },

  server_side_predictor = function(method, datatype, name) {
      return {
        fit: function(data, x_exclude, y_predict, y_exclude, datatypes, distinct, callback, callback_error) {
          // x_exclude, y_predict, y_exclude, scale
          config = {
            method: method,
            'x_exclude': x_exclude, // number of missing cols to exclude
            'y_exclude': JSON.stringify(Array.from(y_exclude)),
            'y_predict': y_predict, // null or number
            'scale': '1',
            'datatype': JSON.stringify(datatypes),
            'distinct': JSON.stringify(distinct)
          }
          process_on_server(config, callback, callback_error);
        },

        datatype: datatype,
        name: name
      }
   },

  ajax_callback = function(callback) {
    return function(data) {
      callback(data['result']);
    }
  },

  process_on_server = function(config, callback, callback_error) {
    $.ajax({
      type: "POST",
      url: g['url'],
      data: config,
      success: ajax_callback(callback),
      error: callback_error,
      dataType: 'json'
    });
  };
