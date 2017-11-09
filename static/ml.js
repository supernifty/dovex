
var
  // example supervised learning algorithm
  // TODO confusion
  ml = {
    // categorical prediction
    'majority': function() {
      var param, xcols, ycol;
      return {
        fit: function(data, xcols_in, ycol_in) {
          ycol = ycol_in;
          xcols = xcols_in;
          yvals = numeric.transpose(data)[ycol]; 
          param = math.mode(yvals)[0];
        },

        score: function(data) {
          correct = 0;
          for (row in data) {
            if (data[row][ycol] == param) {
              correct += 1;
            } 
          }
          return correct;
        },

        datatype: 'categorical',
        name: 'Baseline (majority class)',
      }
    },
    // numeric prediction
    'mean': function() {
      var param, xcols, ycol;
      return {
        fit: function(data, xcols_in, ycol_in) {
          ycol = ycol_in;
          xcols = xcols_in;
          yvals = numeric.transpose(data)[ycol]; 
          param = math.mean(yvals);
        },

        score: function(data) {
          se = 0;
          for (row in data) {
            se += Math.pow(data[row][ycol] - param, 2);
          }
          return Math.sqrt(se / data.length);
        },
        datatype: 'numeric',
        name: 'Baseline (mean)',
      }
    }
  };
