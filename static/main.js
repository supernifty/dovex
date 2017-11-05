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
    var summary = {}, 
      converted = [];

    for (header in data['meta']['header']) { // 0..len
      summary[header] = {'missing': 0, 'distinct': new Set()}
    }
    // populate
    for (row in data['data']) {
      for (col in data['data'][row]) { // 0..col
        if (data['data'][row][col] == '') {
          summary[col]['missing'] += 1;
        }
        else {
          summary[col]['distinct'].add(data['data'][row][col]);
        }
      }
    }

    // convert to datatable
    for (column in data['meta']['header']) { // 0..len
      converted.push([data['meta']['header'][column], 100 * summary[column]['missing'] / data['data'].length, summary[column]['distinct'].size])
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
            console.log(data);
            return Math.round(data);
          }
        } ],
      "data": converted
    });
  },
  
  process = function(data) {
    show_overview(data);
    show_columns(data);
  };
