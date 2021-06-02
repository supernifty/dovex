var
  g = {}, // globals

  set_update = function(msg) {
    $('#summary').attr('class', 'alert alert-info');
    $('#summary').html(msg);
  },

  set_url = function(url) {
    g['url'] = url;
  },

  set_error = function() {
    $('#summary').attr('class', 'alert alert-danger alert-dismissable');
    $('#summary').html('Failed to load data.');
  },

  // NB we now assume datatype will be provided in meta, and don't calculate it
  calculate_summary = function() {
    var summary = {'columns': {}, 'missing_row': []},
      missing_row;

    for (header in g['data']['meta']['header']) { // 0..len
      summary['columns'][header] = {'missing': 0, 'distinct': {}, 'min': 1e9, 'max': -1e9, 'count': 0, 'sum': 0};
    }

    // populate summary - missing_row is a list of how many columns are missing for each row
    // excluding rows marked as to exclude
    g['max_missing'] = 0;
    for (row in g['data']['data']) {
      missing_row = 0;
      for (col in g['data']['data'][row]) { // 0..col
        if (g['data']['data'][row][col] == '') {
          //datatype[col] = 'categorical'; // TODO missing data is automatically categorical (for now)
          summary['columns'][col]['missing'] += 1;
          if (!g['excluded_cols'].has(parseInt(col))) { // only count if not excluded
            missing_row += 1;
          }
        }
        else { // not missing
          if (!$.isNumeric(g['data']['data'][row][col])) {
            //datatype[col] = 'categorical';
          }
          else {
            summary['columns'][col]['min'] = Math.min(summary['columns'][col]['min'], g['data']['data'][row][col]);
            summary['columns'][col]['max'] = Math.max(summary['columns'][col]['max'], g['data']['data'][row][col]);
            summary['columns'][col]['count'] += 1;
            summary['columns'][col]['sum'] += Number(g['data']['data'][row][col]);
          }
        }
        if (!(g['data']['data'][row][col] in summary['columns'][col]['distinct'])) {
          summary['columns'][col]['distinct'][g['data']['data'][row][col]] = 0;
        }
        summary['columns'][col]['distinct'][g['data']['data'][row][col]] += 1;
      }
      summary['missing_row'].push(missing_row);
      g['max_missing'] = Math.max(g['max_missing'], missing_row);
    }

    g['summary'] = summary;
    show_max_missing();
  },

  show_overview = function() {
    set_update('Loaded <strong>' + g['data']['data'].length + '</strong> rows with <strong>' + g['data']['meta']['header'].length + '</strong> columns.');
  },

  show_columns = function() {
    // prepare summary object
    var converted = [];

    // convert to datatable
    for (column in g['data']['meta']['header']) { // 0..len
      if (g['data']['meta']['datatype'][column] != 'categorical' && g['summary']['columns'][column]['count'] > 0) {
        converted.push([
          g['data']['meta']['header'][column],
          100 * g['summary']['columns'][column]['missing'] / g['data']['data'].length,
          Object.keys(g['summary']['columns'][column]['distinct']).length,
          g['data']['meta']['datatype'][column],
          g['summary']['columns'][column]['min'],
          g['summary']['columns'][column]['max'],
          (g['summary']['columns'][column]['sum'] / g['summary']['columns'][column]['count']).toFixed(1)
        ]);
      }
      else {
        converted.push([
          g['data']['meta']['header'][column],
          100 * g['summary']['columns'][column]['missing'] / g['data']['data'].length,
          Object.keys(g['summary']['columns'][column]['distinct']).length,
          g['data']['meta']['datatype'][column],
          '',
          '',
          ''
        ]);
      }
    }

    $('#table_columns tbody').off('click', 'tr');
    $('#table_columns').DataTable({
      "destroy": true,
      "order": [[ 0, "asc" ]],
      "paging": false,
      "searching": false,
      "bInfo" : false,
      "columnDefs": [ {
        "targets": 1, //  missing%
          "render": function ( data, type, full, meta ) {
            return data.toFixed(1);
          }
        }
      ],
      "select": {
        style: 'os',
        selector: 'td:first-child'
      },
      "fnRowCallback": function(nRow, aData, iDisplayIndex, iDisplayIndexFull ) {
        if (g["excluded_cols"].has(g['data']['meta']['header'].indexOf(aData[0]))) {
          $(nRow).addClass("excluded");
        }
        return nRow;
      },
      "data": converted
    });
    $('#table_columns').width($('.container').width());
    $('#table_columns tbody').on('click', 'tr', select_overview);
  },

  show_missing = function() {
    var
      layout = { title: '% Missing data for each column', xaxis: {}, barmode: 'stack' },
      x = [],
      y_out = [],
      y_in = [];

    for (column in g['summary']['columns']) {
      if (g['excluded_cols'].has(parseInt(column))) {
        y_out.push(100 * g['summary']['columns'][column]['missing'] / g['data']['data'].length);
        y_in.push(0);
      }
      else {
        y_out.push(0);
        y_in.push(100 * g['summary']['columns'][column]['missing'] / g['data']['data'].length);
      }
      x.push(g['data']['meta']['header'][column]);
    }

    converted = [ {
       x: x,
       y: y_in,
       type: 'bar',
       name: 'Included',
       color: 'blue'
    }, {
       x: x,
       y: y_out,
       type: 'bar',
       name: 'Excluded',
       color: 'red'
    } ];

    Plotly.purge(document.getElementById("missing_by_column"));
    Plotly.plot("missing_by_column", converted, layout, {displayModeBar: g['displayModeBar']});

    // rows with missing data
    layout = {
      title: 'Rows with missing data - excluding ' + g['excluded_cols'].size + ' column(s)',
      bargap: 0.05,
      xaxis: { title: 'Number of columns of missing data' },
      yaxis: { title: 'Number of rows' }
    };
    converted = [ { x: g['summary']['missing_row'], type: 'histogram' } ];

    Plotly.purge(document.getElementById("missing_by_row"));
    Plotly.plot("missing_by_row", converted, layout, {displayModeBar: g['displayModeBar']});
  },

  graph_axis = function(tab, col, axis) {
    var
      key = tab + '_' + col + '_' + axis;
    if (g['graph_axis_style'].has(key)) {
      return 'log';
    }
    else {
      return '-';
    }
  },

  update_rel_graph = function(ev) {
    var target = "" + ev.target.id,
      key = 'rel_' + target.split('_')[1] + '_' + target.split('_')[2];
    if (g['graph_axis_style'].has(key)) {
      g['graph_axis_style'].delete(key);
    }
    else {
      g['graph_axis_style'].add(key);
    }
    show_relationships();
  },

  update_dist_graph = function(ev) {
    var target = "" + ev.target.id,
      key = 'dist_' + target.split('_', 2)[1] + '_' + target.split('_')[2];
    if (g['graph_axis_style'].has(key)) {
      g['graph_axis_style'].delete(key);
    }
    else {
      g['graph_axis_style'].add(key);
    }
    show_column_dists();
  },

  graph_dropdown = function(tab, col, axis) {
    var
      key = tab + '_' + col + '_' + axis;
    // we are not including the axis since it is always y
    if (g['graph_axis_style'].has(key)) {
      return "<li><a id='" + tab + "_" + col + "_" + axis + "' href='#'>Use linear scale for " + escape_html(g['data']['meta']['header'][col]) + "</a></li>";
    }
    else {
      return "<li><a id='" + tab + "_" + col + "_" + axis + "' href='#'>Use log scale for " + escape_html(g['data']['meta']['header'][col]) + "</a></li>";
    }
  },

  sorted_keys = function(d) {
    var result = [];
    for(var key in d) {
      result[result.length] = key;
    }
    result.sort();
    return result;
  },

  show_column_dists = function() {
    const
      COLS_PER_GRAPH = 6;
    var
      cols = numeric.transpose(g['data']['data']),
      converted, x, y, layout,
      width = Math.round(COLS_PER_GRAPH/12 * $('.container').width()),
      exclude_missing = $('#distribution_missing').prop('checked'),
      log_axes_list = '';
    $('#distributions').empty();
    for (var col in g['data']['meta']['header']) { // 0.. cols
      if (g['excluded_cols'].has(parseInt(col))) {
        continue;
      }
      x = [];
      y = [];
      if (g['data']['meta']['datatype'][col] == 'categorical') {
        // sort the keys
        for (var distinct of sorted_keys(g['summary']['columns'][col]['distinct'])) {
           if (exclude_missing && (distinct == '' || g['summary']['columns'][col]['distinct'][distinct] == '')) {
             continue;
           }
           x.push(distinct);
           y.push(g['summary']['columns'][col]['distinct'][distinct]);
        }
        converted = [ {
           x: x,
           y: y,
           type: 'bar'
        } ];
        layout = { title: g['data']['meta']['header'][col], xaxis: { title: g['data']['meta']['header'][col], type: 'category'}, yaxis: { title: 'Count', type: graph_axis('dist', col, 'y') }, margin: { r: 0, pad: 0 } };
      }
      else { // numeric
        converted = [{ x: cols[col], type: 'histogram' }];
        layout = { title: g['data']['meta']['header'][col], xaxis: { title: g['data']['meta']['header'][col], type: graph_axis('dist', col, 'x')}, yaxis: { title: 'Count', type: graph_axis('dist', col, 'y') }, margin: { r: 0, pad: 0 } };
        // log_axes_list += "<li><a id='dist_" + col + "_x' href='#'>" + g['data']['meta']['header'][col] + ": x-axis</a></li>"; // plot.ly bug
      }
      log_axes_list += graph_dropdown('dist', col, 'y'); 
      target = $('#distributions').append('<div class="col-md-' + COLS_PER_GRAPH + '"><div id="dist_' + col + '" style="width: ' + width + 'px"></div></div>');
      Plotly.purge(document.getElementById("dist_" + col));
      Plotly.plot("dist_" + col, converted, layout, {displayModeBar: g['displayModeBar']});
    }
    $('#distribution_log_ul').html(log_axes_list);
  },

  init_relationships = function() {
    $('#relationship_feature').empty();
    for (header in g['data']['meta']['header']) {
      if (g['excluded_cols'].has(parseInt(header))) {
        continue;
      }
      $('#relationship_feature').append($('<option>', {value:header, text:g['data']['meta']['header'][header]}));
    }
    $('#relationship_label').empty();
    $('#relationship_label').append($('<option>', {value:'', text:'(none)'}));
    for (header in g['data']['meta']['header']) {
      $('#relationship_label').append($('<option>', {value:header, text:g['data']['meta']['header'][header]}));
    }
  },

  show_relationships = function() {
    const
      COLS_PER_GRAPH = 6,
      MAX_CATEGORIES = 100;
    var
      cols = numeric.transpose(g['data']['data']),
      feature = $('#relationship_feature').val(),
      label = $('#relationship_label').val(),
      converted, x, y, layout,
      width = Math.round(COLS_PER_GRAPH/12 * $('.container').width()),
      exclude_missing = $('#relationship_missing').prop('checked'),
      log_axes_list = '', x_vals;

    // clear existing plots if any
    $('#relationships div div').each(function () {
        if (this.id != '' && this.id.indexOf('corr_') == 0) {
          Plotly.purge(this.id);
        }
    });
    $('#relationships').empty();

    // check not too many categories (plot.ly can't handle)
    distinct_count = Object.keys(g['summary']['columns'][feature]['distinct']).length;
    if (g['data']['meta']['datatype'][feature] == 'categorical' && distinct_count > MAX_CATEGORIES) {
      $('#relationships').html('<div class="alert alert-danger fade in">This feature has too many categories (<strong>' + distinct_count + '</strong>&gt;' + MAX_CATEGORIES + ')</div>');
      return;
    }

    for (var col in g['data']['meta']['header']) { // 0.. cols
      if (col == feature) {
        continue;
      }
      if (g['excluded_cols'].has(parseInt(col))) {
        continue;
      }
      // cat vs cat -> stacked bar
      if (g['data']['meta']['datatype'][col] == 'categorical' && g['data']['meta']['datatype'][feature] == 'categorical') {
        counts = {};
        x_vals = new Set();
        for (row in g['data']['data']) {
          feature_val = g['data']['data'][row][feature];
          x_val = g['data']['data'][row][col];

          if (exclude_missing && (feature_val == '' || x_val == '')) {
            continue;
          }

          x_vals.add(x_val)
          if (!(feature_val in counts)) {
            counts[feature_val] = {}
          }
          if (!(x_val in counts[feature_val])) {
            counts[feature_val][x_val] = 0
          }
          counts[feature_val][x_val] += 1
        }
        // convert to traces - one trace per feature
        converted = [];
        xvals_sorted = Array.from(x_vals).sort();
        for (var current_feature_val in counts) {
          x = [];
          y = [];
          for (current_x_val of xvals_sorted) {
            if (current_x_val.startsWith('_dx')) {
              x.push(current_x_val.slice(4)); // hack for sorting
            }
            else {
              x.push(current_x_val);
            }
            y.push(counts[current_feature_val][current_x_val]);
          }
          if (current_feature_val.startsWith('_dx')) {
            current_feature_val = current_feature_val.slice(4);
          }
          converted.push({
            x: x,
            y: y,
            name: current_feature_val,
            type: 'bar'
          });
        }
        layout = { title: g['data']['meta']['header'][col], xaxis: { type: 'category' }, yaxis: { type: graph_axis('rel', col, 'y'), title: 'Count' }, margin: { r: 0, pad: 0 }, barmode: 'stack' }; // , height: 800, width: 1200 };
        log_axes_list += graph_dropdown('rel', col, 'y');
      }
      else if (g['data']['meta']['datatype'][col] != 'categorical' && g['data']['meta']['datatype'][feature] != 'categorical') { // num vs num -> scatter
        x = [];
        y = [];
        z = [];
        c = [];
        seen = [];
        for (row in g['data']['data']) {
          x.push(g['data']['data'][row][col])
          y.push(g['data']['data'][row][feature])
          // selected column as label
          if (label != '') {
            z.push(g['data']['data'][row][label]);
            if (g['data']['meta']['datatype'][label] != 'categorical') {
              c.push(g['colours'][0]); // TODO range not yet implemented
            }
            else {
              if (seen.indexOf(z[z.length - 1]) == -1) {
                seen.push(z[z.length - 1]);
              }
              c.push(g['colours'][seen.indexOf(z[z.length - 1]) % g['colours'].length]);
            }
          }
          else {
            z.push('');
            c.push(g['colours'][0]);
          }
        }
        converted = [{
          x: x,
          y: y,
          text: z,
          mode: 'markers',
          type: 'scatter',
          opacity: 0.8,
          marker: { 'color': c, 'size': MARKER_SIZE }
        }];
        layout = { title: g['data']['meta']['header'][col], xaxis: { title: g['data']['meta']['header'][col], type: graph_axis('rel', col, 'x') }, yaxis: { title: g['data']['meta']['header'][feature], type: graph_axis('rel', col, 'y') }, margin: { r: 0, pad: 0 }, barmode: 'stack', hovermode: 'closest' }; // , height: 800, width: 1200 };
        // log_axes_list += "<li><a id='rel_" + col + "_x' href='#'>" + g['data']['meta']['header'][col] + ": x-axis</a></li>"; // plot.ly bug
        log_axes_list += graph_dropdown('rel', col, 'y'); 
      }
      else if (g['data']['meta']['datatype'][col] != 'categorical' && g['data']['meta']['datatype'][feature] == 'categorical') { // col-numeric (x) vs feature-cat (y)
        counts = {};
        max_feature_val = g['summary']['columns'][col]['max'];
        // plot.ly doesn't like large values, so scale, based on max
        scale = '';
        scale_factor = 0;

        while (max_feature_val / Math.pow(10, scale_factor) > 100) { // need to scale
          scale_factor += 1;
          scale = ' (&times;10<sup>' + scale_factor + '</sup>)';
        }

        for (row in g['data']['data']) {
          feature_val = g['data']['data'][row][feature];
          x_val = g['data']['data'][row][col];

          if (exclude_missing && (feature_val == '' || x_val == '')) {
            continue;
          }

          if (!(feature_val in counts)) {
            counts[feature_val] = [];
          }
          if ($.isNumeric(x_val)) { // plot.ly doesn't like empty values
            counts[feature_val].push(x_val / Math.pow(10, scale_factor));
          }
        }

        // convert to traces - one trace per feature
        converted = [];
        for (var current_feature_val of sorted_keys(counts)) {
          if (counts[current_feature_val].length > 0) {
            if (current_feature_val.startsWith('_dx')) {
              label = current_feature_val.slice(4);
            }
            else {
              label = current_feature_val;
            }
            converted.push({
              x: counts[current_feature_val],
              name: label,
              type: 'histogram'
            });
          }
        }
        layout = { title: g['data']['meta']['header'][col], xaxis: { title: g['data']['meta']['header'][col] + scale, type: graph_axis('rel', col, 'x') }, yaxis: { title: 'Count', type: graph_axis('rel', col, 'y') }, margin: { r: 0, pad: 0 }, barmode: 'stack' }; // , height: 800, width: 1200};
        // log_axes_list += "<li><a id='rel_" + col + "_x' href='#'>" + g['data']['meta']['header'][col] + ": x-axis</a></li>"; // plot.ly bug
        log_axes_list += graph_dropdown('rel', col, 'y');
      }
      else { // cat (x) vs num (y)
        col_distinct_count = Object.keys(g['summary']['columns'][col]['distinct']).length;
        if (col_distinct_count > MAX_CATEGORIES) {
          // TODO tell user
          continue;
        }
        counts = {}
        x_vals = new Set();
        for (row in g['data']['data']) {
          feature_val = g['data']['data'][row][feature];
          x_val = g['data']['data'][row][col];

          if (exclude_missing && (feature_val == '' || x_val == '')) {
            continue;
          }

          x_vals.add(x_val)
          if (!(x_val in counts)) {
            counts[x_val] = []
          }
          counts[x_val].push(feature_val)
        }
        // convert to traces - one trace per feature
        converted = [];
        xvals_sorted = Array.from(x_vals).sort();
        for (var current_x_val of xvals_sorted) {
           if (current_x_val.startsWith('_dx')) {
             x = current_x_val.slice(4); // hack for sorting
           }
           else {
             x = current_x_val;
           }
           converted.push({
            y: counts[current_x_val],
            name: x,
            type: 'box',
            boxpoints: 'Outliers'
          });
        }
        layout = { title: g['data']['meta']['header'][col], xaxis: { type: 'category', title: g['data']['meta']['header'][col]}, yaxis: { title: g['data']['meta']['header'][feature], type: graph_axis('rel', col, 'y') }, margin: { r: 0, pad: 0 } }; // , height: 800, width: 1200};
        log_axes_list += graph_dropdown('rel', col, 'y');
      }
      $('#relationships').append('<div class="col-md-' + COLS_PER_GRAPH + '"><div id="corr_' + col + '" style="width: ' + width + 'px"></div></div>');
      try {
        Plotly.plot("corr_" + col, converted, layout, {displayModeBar: g['displayModeBar']});
      } catch (error) {
        $('#corr_' + col).html('Error: ' + error);
      }
    }
    $('#relationships_log_ul').html(log_axes_list);
  },

  init_prediction = function() {
    var old_header = $('#outcome').val();
    $('#outcome').empty();
    $('#projection_highlight').empty();
    $('#projection_highlight_2').empty();
    $('#projection_highlight_2').append($('<option>', {value:'', text:'(same)'}));
    for (header in g['data']['meta']['header']) {
      $('#outcome').append($('<option>', {value:header, text:g['data']['meta']['header'][header]}));
      $('#projection_highlight').append($('<option>', {value:header, text:g['data']['meta']['header'][header]}));
      $('#projection_highlight_2').append($('<option>', {value:header, text:g['data']['meta']['header'][header]}));
    }
    if (old_header != null) {
      $('#outcome').val(old_header);
      $('#projection_highlight').val(old_header);
      $('#projection_highlight_2').val(old_header);
    }
    show_predictors();
    update_excluded();
  },

  /* change in excluded columns */
  update_excluded = function() {
    var excluded_list = [];
    for (header in g['data']['meta']['header']) {
      if (g['excluded_cols'].has(parseInt(header))) {
        excluded_list.push(g['data']['meta']['header'][header]);
      }
    }
    if (excluded_list.length == 0) {
      $('#prediction_config').html('<div class="alert alert-info">All inputs will be included in the analysis.</div>');
    }
    else {
      $('#prediction_config').html('<div class="alert alert-info"><strong>Excluded inputs:</strong> ' + excluded_list.join(', ') + '</div>');
    }
    calculate_summary();
    g['missing_ok'] = false;
    g['dists_ok'] = false;
    g['relationships_ok'] = false;
    g['data_ok'] = false;
    g['correlation_ok'] = false;
  },

  show_predictors = function() {
    var outcome_datatype = g['data']['meta']['datatype'][$('#outcome').val()],
      old_predictor = $('#predictor').val();
    $('#predictor').empty();
    for (predictor in ml) {
      if (outcome_datatype == ml[predictor]().datatype) {
        $('#predictor').append('<option value="' + predictor + '">' + ml[predictor]().name + '</option>');
      }
    }
    if (old_predictor != null) {
      $('#predictor').val(old_predictor);
    }
  },

  show_max_missing = function() {
    var old_mm = $('#max_missing').val(),
      old_mmp = $('#max_missing_projection').val();
    $('#max_missing').empty();
    $('#max_missing_projection').empty();
    for (var i = 1; i < g['max_missing']; i++) {
      $('#max_missing').append('<option value="' + i + '">Exclude rows with ' + i + ' or more missing columns</option>');
      $('#max_missing_projection').append('<option value="' + i + '">Exclude rows with ' + i + ' or more missing columns</option>');
    }
    $('#max_missing').append('<option selected="selected" value="' + (g['max_missing'] + 1) + '">Include all rows</option>');
    $('#max_missing_projection').append('<option selected="selected" value="' + (g['max_missing'] + 1) + '">Include all rows</option>');
    if (old_mm != null) {
      $('#max_missing').val(old_mm);
    }
    if (old_mmp != null) {
      $('#max_missing_projection').val(old_mmp);
    }
    // make sure both values set
    if ($('#max_missing').val() == '') {
      $('#max_missing').val(g['max_missing'] + 1);
    }
    if ($('#max_missing_projection').val() == '') {
      $('#max_missing_projection').val(g['max_missing'] + 1);
    }
  }

  show_prediction = function() {
    var predictor = ml[$('#predictor').val()](),
      outcome_datatype = g['data']['meta']['datatype'][$('#outcome').val()],
      distinct = {};
    for (column in g['summary']['columns']) {
      distinct[column] = Object.keys(g['summary']['columns'][column]['distinct']).length;
    }
    $('#prediction_result').html('<div class="alert alert-info">Processing...</div>')
    predictor.fit(g['data']['data'], $('#max_missing').val(), $('#outcome').val(), g['excluded_cols'], g['data']['meta']['datatype'], distinct, '', prediction_result_callback, prediction_result_callback_error, $('#class_weight').val());
  },

  prediction_result_callback_error = function() {
      $('#prediction_result').html('<div class="alert alert-danger alert-dismissable">An error occurred. Prediction failed.</div>')
  },

  plot_prediction_features = function(data, target) {
      // get top 10
      var indices = new Array(data.length), x = [], y = [];
      for (var i = 0; i < indices.length; i++ ) {
        indices[i] = i;
      }
      indices.sort(function(a, b) { return data[a] < data[b] ? 1 : -1; });

      for (var i = 0; i < Math.min(10, indices.length); i++) {
        if (data[indices[i]] == 0) {
          break;
        }
        x.push(g['data']['meta']['header'][indices[i]]);
        y.push(data[indices[i]]);
      }

      layout = { title: 'Feature Importance', xaxis: {} },
      data = [ {
       x: x,
       y: y,
       type: 'bar'
      } ];

      Plotly.plot(target, data, layout, {displayModeBar: g['displayModeBar']});

  }

  prediction_result_callback = function(result) { // training_score, cross_validation_score, predictions) {
    var predictor = ml[$('#predictor').val()](),
      outcome_datatype = g['data']['meta']['datatype'][$('#outcome').val()],
      prediction_name = g['data']['meta']['header'][$('#outcome').val()] + ' - PREDICTIONS',
      notes = '';
    Plotly.purge(document.getElementById('confusion'));
    Plotly.purge(document.getElementById('feature_importance'));
    if ('error' in result) { // problem
      $('#prediction_result').html('<div class="alert alert-danger alert-dismissable">Error: ' + result['error'] + '</div>');
      return;
    }

    if ('notes' in result && result['notes'].length > 0) {
      notes = '<ul>';
      for (item in result['notes']) {
        notes += '<li>' + result['notes'][item] + '</li>';
      }
      notes += '</ul>';
    }
    if (outcome_datatype == 'categorical') { // categorical predictor
      $('#prediction_result').html(
        '<div class="alert alert-info alert-dismissable">' +
        '<strong>Training samples: </strong>' + result['predictions'].length + '<br/>' +
        '<strong>Training accuracy: </strong>' + ((result['training_score'] * 100).toFixed(1)) + '%<br/>' +
        '<strong>Cross validation accuracy: </strong>' + ((result['cross_validation_score'] * 100).toFixed(1)) + '%<br/>' +
        '<br/>Predictions have been added to the dataset as <strong>' + prediction_name + '</strong>' + notes + '</div>');
      // confusion matrix
      if ('confusion' in result) {
        // normalize
        var z = [];
        result['confusion'].forEach(function(el) {
          var zr = [],
            total = math.sum(el);
          el.forEach(function(c) {
            zr.push((100 * c / total).toFixed(1) + '%');
          })
          z.push(zr);
        });

        data = [{
          x: result['y_labels'],
          y: result['y_labels'],
          z: result['confusion'],
          type: 'heatmap'
        }];
        layout = {title: 'Confusion Matrix', xaxis: {title: 'Predicted Class', type: 'category'}, yaxis: {title: 'Actual Class', type: 'category'}, annotations: []};
        // show values
        for ( var i = 0; i < result['y_labels'].length; i++ ) {
          for ( var j = 0; j < result['y_labels'].length; j++ ) {
            var annotation = {
              xref: 'x1',
              yref: 'y1',
              x: result['y_labels'][j],
              y: result['y_labels'][i],
              text: z[i][j],
              showarrow: false,
              font: {
                color: 'white'
              }
            }
            layout.annotations.push(annotation);
          }
        }
        Plotly.plot("confusion", data, layout, {displayModeBar: g['displayModeBar']});
      }
      else {
        // $('#confusion').html('Confusion matrix is not available');
      }
    }
    else { // numeric predictor
      $('#prediction_result').html(
        '<div class="alert alert-info alert-dismissable"><strong>Training R<sup>2</sup>: </strong>' + (result['training_score'].toFixed(2)) + '<br/>' +
        '<strong>Cross validation R<sup>2</sup>: </strong>' + (result['cross_validation_score'].toFixed(2)) +
        '<br/>Predictions have been added to the dataset as <strong>' + prediction_name + '</strong>' + notes + '</div>');
    }

    if ('features' in result) {
      plot_prediction_features(result['features'], 'feature_importance');
   }
    else {
      // $('#feature_importance').html('Feature importance is not available');
    }

    add_predictions_to_dataset(prediction_name, result['predictions']); // update dataset with predictions
  },

  add_predictions_to_dataset = function(prediction_name, predictions) {
    var cols = numeric.transpose(g['data']['data']),
      expanded_predictions = [],
      max_missing = parseInt($('#max_missing').val());

    if (g['has_predictions']) {
      cols.pop(); // remove existing data column
      g['data']['meta']['header'].pop();
      g['data']['meta']['datatype'].pop();
    }
    g['excluded_cols'].add(cols.length);
    for (var i = 0; i < cols[0].length; i++) {
      if (g['summary']['missing_row'][i] >= max_missing) {
        expanded_predictions.push('');
      }
      else {
        expanded_predictions.push(predictions.shift());
      }
    }
    cols.push(expanded_predictions);
    g['data']['data'] = numeric.transpose(cols);
    g['data']['meta']['header'].push(prediction_name);
    g['data']['meta']['datatype'].push(g['data']['meta']['datatype'][$('#outcome').val()]);
    g['has_predictions'] = true;
    // recalculate
    calculate_all();
  },

  change_reducer = function() {
    var reducer = $('#reducer').val();
    if (reducer == 'tsne') {
      $('#cluster_custom').html('<div class="col-md-4">Perplexity: <input class="form-control" type="text" value="30" id="perplexity"></div>');
    }
    else {
      $('#cluster_custom').html('');
    }
  },

  show_reduction = function() {
    var reducer = ml[$('#reducer').val()](),
      distinct = {};
    for (column in g['summary']['columns']) {
      distinct[column] = Object.keys(g['summary']['columns'][column]['distinct']).length;
    }
    $('#reduction_result').html('<div class="alert alert-info">Processing...</div>')
    reducer.fit(g['data']['data'], $('#max_missing_projection').val(), null, g['excluded_cols'], g['data']['meta']['datatype'], distinct, $('#perplexity').val(), reduction_result_callback, reduction_result_callback_error, '');
  },

  projection_feature = function(data, target, component) {
      // get top 10
      var indices = new Array(data.length), x = [], y = [];
      for (var i = 0; i < indices.length; i++ ) {
        indices[i] = i;
      }
      indices.sort(function(a, b) { return data[a] < data[b] ? 1 : -1; });

      for (var i = 0; i < Math.min(10, indices.length); i++) {
        if (data[indices[i]] == 0) {
          break;
        }
        x.push(g['data']['meta']['header'][indices[i]]);
        y.push(data[indices[i]]);
      }

      layout = { title: 'Feature Importance (component ' + component + ')', xaxis: {} },
      converted = [ {
       x: x,
       y: y,
       type: 'bar'
      } ];

      Plotly.plot(target, converted, layout, {displayModeBar: g['displayModeBar']});
  },

  reduction_result_callback = function(result) {
    var traces = {},
        converted = [],
        layout,
        datatype_highlight = g['data']['meta']['datatype'][$('#projection_highlight').val()],
        highlight_col = $('#projection_highlight').val(),
        graph_type = $('#projection_text').prop('checked') ? "markers+text" : "markers",
        max_missing = parseInt($('#max_missing_projection').val()),
        width = Math.round($('.container').width()),
        point = 0,
        notes = '',
        highlight_value;

    if ('error' in result) {
      $('#reduction_result').html('<div class="alert alert-danger alert-dismissable">An error occurred: ' + result['error'] + '</div>')
      return;
    }
    Plotly.purge(document.getElementById('reduction'));
    Plotly.purge(document.getElementById('projection_features'));
    Plotly.purge(document.getElementById('projection_features_2'));
    for (var i=0; i < g['data']['data'].length; i++) {
      if (g['summary']['missing_row'][i] >= max_missing) {
        continue
      }
      highlight_value = g['data']['data'][i][highlight_col];
      if (datatype_highlight == 'categorical') {
        if (!(highlight_value in traces)) {
          traces[highlight_value] = {'x': [], 'y': [], 'text': [], textposition: 'bottom right', name: highlight_value, mode: graph_type, type: 'scatter', opacity: 0.8, marker: {size: MARKER_SIZE}};
        }
        traces[highlight_value]['x'].push(result['projection'][point][0]);
        traces[highlight_value]['y'].push(result['projection'][point][1]);
        if ($('#projection_highlight').val() == '') {
          traces[highlight_value]['text'].push(highlight_value);
        } else {
          traces[highlight_value]['text'].push(g['data']['data'][i][$('#projection_highlight_2').val()]);
        }
      }
      else {
        if (!('' in traces)) {
          traces[''] = {'x': [], 'y': [], 'text': [], textposition: 'bottom right', mode: graph_type, type: 'scatter', opacity: 0.8, showscale: true, marker: {color: [], size: MARKER_SIZE}};
        }
        traces['']['x'].push(result['projection'][point][0]);
        traces['']['y'].push(result['projection'][point][1]);
        if ($('#projection_highlight').val() == '') {
          traces['']['text'].push(highlight_value);
        } else {
          traces['']['text'].push(g['data']['data'][i][$('#projection_highlight_2').val()]);
        }
        traces['']['marker']['color'].push(highlight_value);
      }
      point++;
    }
    for (var trace in traces) {
      converted.push(traces[trace]);
    }
    layout = { title: 'Projection', hovermode: 'closest', width: width, height: width * 2/3 };
    Plotly.plot("reduction", converted, layout);
    if ('features' in result) {
      projection_feature(result['features'], "projection_features", 1 );
      projection_feature(result['features_2'], "projection_features_2", 2 );
    }
    if ('notes' in result && result['notes'].length > 0) {
      notes = '<ul>';
      for (item in result['notes']) {
        notes += '<li>' + result['notes'][item] + '</li>';
      }
      notes += '</ul>';
    }
    $('#reduction_result').html('<div class="alert alert-info">' + point + ' data points transformed.' + notes + '</div>')
  },

  reduction_result_callback_error = function() {
      $('#reduction_result').html('<div class="alert alert-danger alert-dismissable">An error occurred. Analysis failed.</div>')
  },

  select_overview = function() {
    if ( $(this).hasClass('excluded') ) {
      $(this).removeClass('excluded');
      column = $('#table_columns').DataTable().row(this)[0][0];
      g['excluded_cols'].delete(column);
    }
    else {
      $(this).addClass('excluded');
      column = $('#table_columns').DataTable().row(this)[0][0];
      g['excluded_cols'].add(column);
    }
    update_excluded();
  },

  show_data = function() {
    if (!g['data_ok']) {
      var converted = g['data']['data'];
      var tableHeaders = '';
      $.each(g['data']['meta']['header'], function(i, val) {
        //TODO if (!g['excluded_cols'].has(parseInt(i))) { // only count if not excluded
          tableHeaders += "<th>" + val + "</th>";
        //}
      });

      $("#table_data").empty();
      $("#table_data").append('<thead><tr>' + tableHeaders + '</tr></thead>');

      $('#table_data').DataTable({
        "destroy": true,
        "order": [[ 0, "asc" ]],
        "bInfo" : false,
        "pageLength": 50,
        "data": converted
      });
      g['data_ok'] = true;
    }
  },

  show_correlation = function() {
    if (!g['correlation_ok']) {
      var server = ml['correlation']();
      $('#correlation').html('<div class="alert alert-info">Please wait while we calculate p-values...</div>')
      server.fit(g['data']['data'], null, null, g['excluded_cols'], g['data']['meta']['datatype'], null, null, correlation_callback, correlation_callback_error, '');
    }
  },

  show_correlation_subgroup = function(covariate1, covariate2) {
      var server = ml['correlation_subgroup']();
      // $('#correlation').html('<div class="alert alert-info">Please wait while we calculate p-values...</div>')
      server.fit(g['data']['data'], covariate1, covariate2, g['excluded_cols'], g['data']['meta']['datatype'], null, null, correlation_subgroup_callback, correlation_callback_error, '');
  }

  correlation_subgroup_callback = function(result) {
    // now generate table
    $('#correlation_detail_modal').modal('show');
    $("#table_correlation_detail").empty();
    $("#table_correlation_detail").append('<thead><th>Category 1</th><th>Category 2</th><th>p-value</th><th>n</th><th>Mean 1</th><th>Mean 2</th><th>Test</th></tr></thead>');

    $('#table_correlation_detail').DataTable({
      "destroy": true,
      "order": [[ 0, "asc" ]],
      "bInfo" : false,
      "pageLength": 50,
      "data": result["result"],
      "columnDefs": [
        { "targets": 0, "render": function ( data, type, full, meta ) { if (data.startsWith('_dx')) { return data.slice(4) } else { return data } } },
        { "targets": 1, "render": function ( data, type, full, meta ) { if (data.startsWith('_dx')) { return data.slice(4) } else { return data } } },
        { "targets": 2, "render": function ( data, type, full, meta ) { return data.toPrecision(3); } },
        { "targets": 4, "render": function ( data, type, full, meta ) { if (data == '-') { return data } else { return data.toPrecision(3); } } },
        { "targets": 5, "render": function ( data, type, full, meta ) { if (data == '-') { return data } else { return data.toPrecision(3); } } }
      ]
    });

    $('#correlation_detail_label').html('Individual comparison between categories');
  
  }
 
  correlation_callback = function(result) {
      var table_data = [];

      $('#correlation').html('')
      Plotly.purge(document.getElementById("correlation"));
      var data = [{ x: result['xs'], y: result['xs'], z: result['zs'], type: 'heatmap', showscale: true, colorscale: 'RdBu' }],
        annotations = [], current;

      for (x in result['xs']) {
        current = [];
        for (y in result['xs']) {
          annotations.push({
            xref: 'x1', yref: 'y1',
            x: result['xs'][x], 
            y: result['xs'][y],
            text: Math.round(result['zs'][x][y] * 100) / 100 + '<br><sup>n=' + result['cs'][x][y] + "</sup>",
            font: { family: 'Arial', size: 12, color: 'rgb(50, 171, 96)' },
            showarrow: false,
            font: { color: 'white' }
          });
          if (x != y) {
            table_data.push([result['xs'][x], result['xs'][y], result['cs'][x][y], result['zs'][x][y], result['ts'][x][y]]);
          }
        }
      }

      layout = {
        title: 'Correlation between features (p-value)',
        xaxis: { ticks: '', showgrid: true },
        yaxis: { ticks: '', showgrid: true },
        width: 130 + g['data']['meta']['header'].length * 44,
        height: 130 + g['data']['meta']['header'].length * 44,
        annotations: annotations
      };
      Plotly.newPlot('correlation', data, layout);

      // now generate table
      $("#table_correlation").empty();
      $("#table_correlation").append('<thead><tr><th>Covariate 1</th><th>Covariate 2</th><th>n</th><th>p-value</th><th>Test</th></tr></thead>');

      $('#table_correlation').DataTable({
        "destroy": true,
        "order": [[ 0, "asc" ]],
        "bInfo" : false,
        "pageLength": 50,
        "data": table_data,
        "columnDefs": [
          { "targets": 3, "render": function ( data, type, full, meta ) { return data.toPrecision(6); } }, 
          { "targets": 4, "render": function(data, type, full, meta) { if (data == 'Chi-square' || data == 'ANOVA') { return data + " <a onclick='show_correlation_subgroup(\"" + full[0] + "\", \"" + full[1] + "\")'>details...</a>"; } else { return data } } }
        ]
      });
 
      g['correlation_ok'] = true;
  },

  correlation_callback_error = function() {
      $('#correlation').html('<div class="alert alert-danger alert-dismissable">An error occurred. Correlation calculation failed.</div>')
  },

  update_dists = function() {
    if (!g['dists_ok']) {
      show_column_dists();
      g['dists_ok'] = true;
    }
  },

  update_missing = function() {
    if (!g['missing_ok']) {
      show_missing();
      g['missing_ok'] = true;
    }
  },

  update_relationships = function() {
    if (!g['relationships_ok']) {
      init_relationships();
      show_relationships();
      g['relationships_ok'] = true;
    }
  },

  run_queue = function() {
    if (g['queue'].length > 0) {
      g['queue'].shift()();
      setTimeout(run_queue, 5);
    }
  },

  calculate_all = function() {
    g['queue'] = [
      show_overview,
      calculate_summary,
      show_columns,
      update_missing,
      update_dists,
      update_relationships, // optional
      init_prediction,
      save_recent
    ];

    run_queue();
  },

  process = function(data) {
    g['data'] = data;
    g['excluded_cols'] = new Set();
    g['graph_axis_style'] = new Set();
    g['has_predictions'] = false;
    g['displayModeBar'] = true; // TODO setting
    g['data_ok'] = false;
    g['relationships_ok'] = false;
    g['correlation_ok'] = false;
    g['colours'] = ["#3366cc", "#dc3912", "#ff9900", "#109618", "#990099", "#0099c6", "#dd4477", "#66aa00", "#b82e2e", "#316395", "#994499", "#22aa99", "#aaaa11", "#6633cc", "#e67300", "#8b0707", "#651067", "#329262", "#5574a6", "#3b3eac"];

    calculate_all();

    $('#distribution_missing').change(show_column_dists);
    $('#distribution_log_ul').on("click", "li", update_dist_graph);
    $('#relationships_log_ul').on("click", "li", update_rel_graph);

    $('#relationship_feature,#relationship_label,#relationship_missing').change(show_relationships);
    $('#outcome').change(show_predictors);
    $('#run_predictor').click(show_prediction);
    $('#run_reducer').click(show_reduction);
    $('#reducer').change(change_reducer);
    $('a[data-toggle="tab"]').on('shown.bs.tab', function (e) {
      var target = $(e.target).attr("href") // activated tab
      if (target == '#tab_data') {
        show_data();
      }
      else if (target == '#tab_correlation') {
        show_correlation();
      }
      else if (target == '#tab_relationship') {
        update_relationships();
      }
      else if (target == '#tab_missing') {
        update_missing();
      }
      else if (target == '#tab_inputs') {
        update_dists();
      }
      else if (target == '#tab_cluster') {
        change_reducer();
      }
    });
  },

  save_recent = function() {
    var val = window.localStorage.getItem("recent");
    if (val == null) {
      val = [];
    }
    else {
      val = JSON.parse(val);
    }

    // remove existing
    var idx = val.length;
    while (idx--) {
      if (val[idx].url == window.location.href) { 
        val.splice(idx, 1);
      }
    }

    val.unshift({'url': window.location.href, 'columns': max_length(g.data.meta.header.join(', '), 120)})
    val = val.slice(0, 10);
    window.localStorage.setItem('recent', JSON.stringify(val));
  },

  max_length = function(s, l) {
    if (s.length > l) {
      return s.substr(0, l) + '...';
    }
    else {
      return s
    }
  },

  ENTITY_MAP = {
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    '"': '&quot;',
    "'": '&#39;',
    '/': '&#x2F;',
    '`': '&#x60;',
    '=': '&#x3D;'
  },

  MARKER_SIZE = 8,

  escape_html = function (string) {
    return String(string).replace(/[&<>"'`=\/]/g, function (s) {
      return ENTITY_MAP[s];
    });
  },

  // upload page functionality
  populate_recent = function () {
    var li = '', val = window.localStorage.getItem("recent");
    if (val != null) {
      val = JSON.parse(val);
      if (val.length > 0) {
        for (item in val) {
          li += '<li><a href="' + val[item].url + '">' + val[item].columns + '</a>';
        }
        $('#recent').html('<strong>Recently Viewed Datasets</strong><ul>' + li + '</ul>');
      }
    }
  };
