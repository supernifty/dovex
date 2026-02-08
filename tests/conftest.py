#!/usr/bin/env python
'''
Shared test fixtures for Dovex test suite
'''

import io
import pytest


@pytest.fixture
def simple_numeric_csv():
    """Returns fresh StringIO with 3x3 numeric data for fast unit tests"""
    def _make():
        return io.StringIO("A,B,C\n1,2,3\n4,5,6\n7,8,9")
    return _make()


@pytest.fixture
def simple_categorical_csv():
    """Returns fresh StringIO with categorical data"""
    def _make():
        return io.StringIO("Color,Size\nRed,Small\nBlue,Large\nRed,Medium")
    return _make()


@pytest.fixture
def mixed_data_csv():
    """Returns fresh StringIO with mixed numeric and categorical data"""
    def _make():
        return io.StringIO("Age,Score,Category\n25,100,A\n30,85,B\n35,90,A\n40,95,B")
    return _make()


@pytest.fixture
def data_with_missing_csv():
    """Returns fresh StringIO with missing values"""
    def _make():
        return io.StringIO("A,B,C\n1,2,3\n4,,6\n7,8,")
    return _make()


@pytest.fixture
def data_with_na_variants_csv():
    """Returns fresh StringIO with various NA representations"""
    def _make():
        return io.StringIO("A,B,C\n1,2,3\nNA,5,6\n7,n/a,9\n10,N/A,12")
    return _make()


@pytest.fixture
def basic_config():
    """Minimal valid config dict for preprocessing"""
    return {
        'y_exclude': '[]',
        'x_exclude': '10',  # Allow up to 10 missing columns (permissive)
        'y_predict': '',
        'datatype': '["numeric","numeric","numeric"]',
        'distinct': '{}'
    }


@pytest.fixture
def categorical_config():
    """Config for categorical data"""
    return {
        'y_exclude': '[]',
        'x_exclude': '10',
        'y_predict': '',
        'datatype': '["categorical","categorical"]',
        'distinct': '{"0": 2, "1": 3}'
    }


@pytest.fixture
def mixed_config():
    """Config for mixed data"""
    return {
        'y_exclude': '[]',
        'x_exclude': '10',
        'y_predict': '',
        'datatype': '["numeric","numeric","categorical"]',
        'distinct': '{"2": 2}'
    }


@pytest.fixture
def classification_config():
    """Config for classification with last column as target"""
    return {
        'y_exclude': '[]',
        'x_exclude': '10',
        'y_predict': '2',
        'datatype': '["numeric","numeric","categorical"]',
        'distinct': '{"2": 2}'
    }


@pytest.fixture
def regression_config():
    """Config for regression with last column as target"""
    return {
        'y_exclude': '[]',
        'x_exclude': '10',
        'y_predict': '2',
        'datatype': '["numeric","numeric","numeric"]',
        'distinct': '{}'
    }


@pytest.fixture
def iris_file_handle():
    """Real iris.data file for integration tests"""
    return open('uploads/iris.data', 'r')


@pytest.fixture
def iris_config():
    """Config for iris.data (4 numeric features, 1 categorical target)"""
    return {
        'y_exclude': '[]',
        'x_exclude': '10',
        'y_predict': '4',
        'datatype': '["numeric","numeric","numeric","numeric","categorical"]',
        'distinct': '{"4": 3}'
    }
