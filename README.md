[![Build Status](https://travis-ci.org/supernifty/dovex.svg?branch=master)](https://travis-ci.org/supernifty/dovex)

# Dovex
A web based tool to quickly provide an interactive overview and enable quick exploration of your dataset.

Use to identify columns of interest, explore correlations between inputs, and find problems in your input data.

Try it out by visiting [dovex.org](http://dovex.org/).

## Local Installation
dovex has been tested on Python 3.

```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

To run the software locally, first install the requirements above, then start the web server:
```
python main.py
```

Next, visit http://127.0.0.1:5000/ and upload your data. The software expects the data to be in CSV or TSV format.

The first line should be column headings, followed by lines of data.

## Running with Docker
The app has been packaged as a Docker container.

```
docker run -p 5000:5000 supernifty/dovex:latest
```

By default, the app stores uploaded datasets in the uploads directory. To persist uploaded datasets, use a mount point when starting the docker instance:
```
docker run -p 5000:5000 -v /your/upload/directory:/app/uploads supernifty/dovex:latest
```

Point your browser at http://127.0.0.1:5000/ to use the application.

## Data types
Datatypes can be specified by starting the second line with '#'.
Possible datatypes are: categorical, ordinal, numeric.

**ordinal is currently unimplemented**

Without specifying data types, the software will attempt to infer the datatype of each column.

## Provided functionality

* missing data by column
* missing data by sample
* explore input distributions
* explore relationships between features
* prediction

## Datasets
We have included the following public domain datasets:
* [Iris](http://archive.ics.uci.edu/ml/datasets/Iris)
* [Forest Fires](http://archive.ics.uci.edu/ml/datasets/Forest+Fires)

## Wishlist for v2
* server side focus
* data set management
* row filtering tools
* command line interface
* think about aws serverless or scalable implementation?
