# Dovex
A web based tool to quickly provide an interactive overview of your dataset.

Use to identify columns of interest, explore correlations between inputs, and problems in your input data.

## Installation

```
pip install -r requirements.txt
```

## Usage

To run the software locally, first install the requirements above, then start the web server:
```
python main.py
```

Next, upload your data. The software expects the data to be in CSV format.

The first line should be column headings, followed by lines of data.

## Data types
Datatypes can be specified by starting the second line with '#'.
Possible datatypes are: categorical, ordinal, numeric.

Without specifying data types, the software will attempt to infer the datatype of each column.

## Provided functionality

* missing data by column
* missing data by sample
* explore input distributions
* explore relationships between features

## TODO
* prediction
* clustering
* prediction
* direct url loading

## Datasets
We have included the following public domain datasets:
* iris.data
