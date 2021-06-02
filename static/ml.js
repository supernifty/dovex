
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
    'pca': function() { return server_side_predictor('pca', 'reduce', 'PCA'); },
    'mds': function() { return server_side_predictor('mds', 'reduce', 'MDS'); },
    'tsne': function() { return server_side_predictor('tsne', 'reduce', 't-SNE'); },

    // correlation
    'correlation': function() { return server_side_predictor('correlation', 'reduce', 'correlation'); },
    'correlation_subgroup': function() { return server_side_predictor('correlation_subgroup', 'reduce', 'correlation_subgroup'); }
  },

  server_side_predictor = function(method, datatype, name) {
      return {
        fit: function(data, x_exclude, y_predict, y_exclude, datatypes, distinct, perplexity, callback, callback_error, class_weight) {
          // x_exclude, y_predict, y_exclude, scale
          config = {
            method: method,
            'x_exclude': x_exclude, // number of missing cols to exclude
            'y_exclude': JSON.stringify(Array.from(y_exclude)),
            'y_predict': y_predict, // null or number
            'scale': '1',
            'datatype': JSON.stringify(datatypes),
            'distinct': JSON.stringify(distinct),
            'perplexity': perplexity,
            'class_weight': class_weight
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
