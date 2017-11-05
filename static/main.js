var
  set_updating = function() {
  },

  set_error = function() {
    alert("error");
  },

  show_overview = function(data) {
    $('#table_overview').DataTable({
      "destroy": true,
      "order": [[ 0, "asc" ]],
      "paging": false,
      "searching": false,
      "bInfo" : false,
      "data": [['Rows', data['data'].length], ['Columns', data['meta']['header'].length]]
    });
  },

  show_columns = function(data) {
    // prepare summary object
    var summary = {'missing_col': {}, 'missing_row': []}, 
      converted = [],
      datatype = [],
      missing_row;

    for (header in data['meta']['header']) { // 0..len
      summary['missing_col'][header] = {'missing': 0, 'distinct': new Set()};
      datatype.push('numeric');
    }
    // populate
    for (row in data['data']) {
      missing_row = 0;
      for (col in data['data'][row]) { // 0..col
        if (data['data'][row][col] == '') {
          summary['missing_col'][col]['missing'] += 1;
          missing_row += 1;
        }
        else {
          summary['missing_col'][col]['distinct'].add(data['data'][row][col]);
          if (!$.isNumeric(data['data'][row][col])) {
            datatype[col] = 'categorical';
          }
        }
      }
      summary['missing_row'].push(missing_row);
    }

    if ('datatype' in data['meta']) {
      datatype = data['meta']['datatype'];
    }

    // convert to datatable
    for (column in data['meta']['header']) { // 0..len
      converted.push([data['meta']['header'][column], 100 * summary['missing_col'][column]['missing'] / data['data'].length, summary['missing_col'][column]['distinct'].size, datatype[column]]);
    }

    $('#table_columns').DataTable({
      "destroy": true,
      "order": [[ 0, "asc" ]],
      "paging": false,
      "searching": false,
      "bInfo" : false,
      "columnDefs": [
        {
          "targets": 1,
          "render": function ( data, type, full, meta ) {
            return Math.round(data);
          }
        } ],
      "data": converted
    });

    return summary;
  },

  show_missing = function(summary, data) {
    var
      layout = { title: '% Missing data for each column' },
      x = [], 
      y = [];
    
    for (column in summary['missing_col']) {
      x.push(data['meta']['header'][column]);
      y.push(100 * summary['missing_col'][column]['missing'] / data['data'].length);
    }

    converted = [ {
       x: x,
       y: y,
       type: 'bar'
    } ];

    Plotly.plot("missing_by_column", converted, layout, {displayModeBar: false});

    // rows with missing data
    layout = { 
      title: 'Rows with missing data',
      bargap: 0.05,
      xaxis: { title: 'Number of columns of missing data' },
      yaxis: { title: 'Number of rows' }
    };
    converted = [ { x: summary['missing_row'], type: 'histogram' } ];
    
    Plotly.plot("missing_by_row", converted, layout, {displayModeBar: false});
  },
  
  show_column_dists = function(data) {
  },
  
  show_correlations = function(data) {
  },
  
  show_mds = function(data) {
  },
  
  process = function(data) {
    show_overview(data);
    summary = show_columns(data);
    show_missing(summary, data);
    show_column_dists(data);
    show_correlations(data);
    show_mds(data);
  };
