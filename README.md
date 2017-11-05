# Dover
A web based tool to quickly provide an overview of your data-set.

Use to identify columns of interest, correlations between inputs, and
Fast and interactive web based data-set overview and explorer

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


## Provided functionality

* missing data by column

## TODO
* missing data by sample

