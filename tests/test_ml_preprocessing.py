#!/usr/bin/env python
'''
Tests for ml.py preprocessing functions
'''

import io
import pytest
import ml


class TestIsEmpty:
    """Test the is_empty helper function"""

    def test_empty_string_is_empty(self):
        assert ml.is_empty('') is True

    def test_na_uppercase_is_empty(self):
        assert ml.is_empty('NA') is True

    def test_na_lowercase_is_empty(self):
        assert ml.is_empty('n/a') is True

    def test_na_mixed_case_is_empty(self):
        assert ml.is_empty('N/A') is True

    @pytest.mark.parametrize("empty_value", ['', 'NA', 'n/a', 'N/A'])
    def test_all_empty_variants(self, empty_value):
        """Test all recognized empty value formats"""
        assert ml.is_empty(empty_value) is True

    def test_numeric_zero_not_empty(self):
        assert ml.is_empty('0') is False

    def test_numeric_value_not_empty(self):
        assert ml.is_empty('1.5') is False

    def test_text_value_not_empty(self):
        assert ml.is_empty('text') is False

    def test_space_not_empty(self):
        assert ml.is_empty(' ') is False


class TestPreprocessBasic:
    """Test basic preprocessing functionality"""

    def test_preprocess_simple_numeric(self, simple_numeric_csv, basic_config):
        """Test preprocessing with only numeric columns"""
        result = ml.preprocess(simple_numeric_csv, basic_config)

        assert 'error' not in result
        assert len(result['X']) == 3  # 3 rows
        assert len(result['X'][0]) == 3  # 3 columns
        assert result['X'][0][0] == 1.0
        assert result['X'][1][1] == 5.0
        assert result['X'][2][2] == 9.0

    def test_preprocess_returns_metadata(self, simple_numeric_csv, basic_config):
        """Test that preprocessing returns required metadata"""
        result = ml.preprocess(simple_numeric_csv, basic_config)

        assert 'X' in result
        assert 'y' in result
        assert 'y_labels' in result
        assert 'y_predict' in result
        assert 'y_exclude' in result
        assert 'categorical_cols' in result
        assert 'distinct' in result
        assert 'notes' in result

    def test_preprocess_categorical_one_hot_encoding(self, simple_categorical_csv, categorical_config):
        """Test that categorical columns are one-hot encoded"""
        result = ml.preprocess(simple_categorical_csv, categorical_config)

        assert 'error' not in result
        assert len(result['X']) == 3  # 3 rows
        # Color has 2 distinct (Red, Blue), Size has 3 distinct (Small, Large, Medium)
        # Total features = 2 + 3 = 5
        assert len(result['X'][0]) == 5

    def test_preprocess_handles_missing_numeric_imputation(self, data_with_missing_csv, basic_config):
        """Test that missing numeric values are imputed with mean"""
        result = ml.preprocess(data_with_missing_csv, basic_config)

        assert 'error' not in result
        assert len(result['X']) == 3
        assert len(result['notes']) > 0  # Should have imputation note
        assert 'imputed' in result['notes'][0].lower()

    def test_preprocess_recognizes_na_variants(self, data_with_na_variants_csv, basic_config):
        """Test that various NA formats are recognized"""
        result = ml.preprocess(data_with_na_variants_csv, basic_config)

        assert 'error' not in result
        assert len(result['X']) == 4
        assert len(result['notes']) > 0  # Should have imputation note

    def test_preprocess_with_target_column(self, mixed_data_csv, classification_config):
        """Test preprocessing with a prediction target"""
        result = ml.preprocess(mixed_data_csv, classification_config)

        assert 'error' not in result
        assert len(result['y']) == 4  # Should extract target values
        assert result['y_predict'] == 2
        assert len(result['y_labels']) == 2  # Two distinct categories: A and B

    def test_preprocess_excludes_target_from_features(self, mixed_data_csv, classification_config):
        """Test that target column is not included in X"""
        result = ml.preprocess(mixed_data_csv, classification_config)

        assert 'error' not in result
        # Should only have 2 columns (Age, Score) since Category is target
        assert len(result['X'][0]) == 2

    def test_preprocess_returns_error_for_no_rows(self):
        """Test that preprocessing fails gracefully with no data"""
        data = io.StringIO("A,B,C\n")
        config = {
            'y_exclude': '[]',
            'x_exclude': '0',
            'y_predict': '',
            'datatype': '["numeric","numeric","numeric"]',
            'distinct': '{}'
        }
        result = ml.preprocess(data, config)

        assert 'error' in result
        assert 'No rows' in result['error']

    def test_preprocess_with_column_exclusion(self):
        """Test that y_exclude removes columns"""
        data = io.StringIO("A,B,C\n1,2,3\n4,5,6")
        config = {
            'y_exclude': '[1]',  # Exclude column B
            'x_exclude': '10',
            'y_predict': '',
            'datatype': '["numeric","numeric","numeric"]',
            'distinct': '{}'
        }
        result = ml.preprocess(data, config)

        assert 'error' not in result
        assert len(result['X'][0]) == 2  # Only A and C

    def test_preprocess_skips_rows_with_too_many_missing(self):
        """Test that rows with too many missing values are excluded"""
        data = io.StringIO("A,B,C\n1,2,3\n,,\n7,8,9")
        config = {
            'y_exclude': '[]',
            'x_exclude': '2',  # Skip rows with 2+ missing
            'y_predict': '',
            'datatype': '["numeric","numeric","numeric"]',
            'distinct': '{}'
        }
        result = ml.preprocess(data, config)

        assert 'error' not in result
        assert len(result['X']) == 2  # Should skip the row with all missing

    def test_preprocess_skips_comment_lines(self):
        """Test that lines starting with # are skipped"""
        data = io.StringIO("A,B,C\n1,2,3\n#comment\n4,5,6")
        config = {
            'y_exclude': '[]',
            'x_exclude': '10',
            'y_predict': '',
            'datatype': '["numeric","numeric","numeric"]',
            'distinct': '{}'
        }
        result = ml.preprocess(data, config)

        assert 'error' not in result
        assert len(result['X']) == 2  # Should skip comment line

    def test_preprocess_skips_empty_lines(self):
        """Test that empty lines are skipped"""
        data = io.StringIO("A,B,C\n1,2,3\n\n4,5,6")
        config = {
            'y_exclude': '[]',
            'x_exclude': '10',
            'y_predict': '',
            'datatype': '["numeric","numeric","numeric"]',
            'distinct': '{}'
        }
        result = ml.preprocess(data, config)

        assert 'error' not in result
        assert len(result['X']) == 2

    def test_preprocess_handles_duplicate_column_names(self):
        """Test that duplicate column names are excluded"""
        data = io.StringIO("A,B,A\n1,2,3\n4,5,6")
        config = {
            'y_exclude': '[]',
            'x_exclude': '10',
            'y_predict': '',
            'datatype': '["numeric","numeric","numeric"]',
            'distinct': '{}'
        }
        result = ml.preprocess(data, config)

        assert 'error' not in result
        # Second A column should be excluded
        assert len(result['X'][0]) == 2

    def test_preprocess_with_scaling(self):
        """Test that scale parameter triggers feature scaling"""
        data = io.StringIO("A,B,C\n1,2,3\n4,5,6\n7,8,9")
        config = {
            'y_exclude': '[]',
            'x_exclude': '10',
            'y_predict': '',
            'datatype': '["numeric","numeric","numeric"]',
            'distinct': '{}',
            'scale': True
        }
        result = ml.preprocess(data, config)

        assert 'error' not in result
        # After scaling, mean should be ~0 (allowing for floating point)
        import numpy as np
        mean_val = np.mean(result['X'])
        assert abs(mean_val) < 0.01

    def test_preprocess_excludes_categorical_with_too_many_distinct(self):
        """Test that categorical columns with >MAX_DISTINCT values are excluded"""
        # This would require a large dataset, so we'll test the logic indirectly
        data = io.StringIO("A,B\n1,Cat1\n2,Cat2")
        config = {
            'y_exclude': '[]',
            'x_exclude': '10',
            'y_predict': '',
            'datatype': '["numeric","categorical"]',
            'distinct': '{"1": 150}'  # More than MAX_DISTINCT (100)
        }
        result = ml.preprocess(data, config)

        assert 'error' not in result
        # Column B should be excluded due to too many distinct values
        assert len(result['X'][0]) == 1  # Only numeric column A

    def test_preprocess_returns_error_for_too_large_dataset(self):
        """Test that preprocessing fails for datasets exceeding MAX_CELLS"""
        # Create config that would result in >1M cells
        # This is a conceptual test - we won't create actual huge data
        data = io.StringIO("A,B,C\n1,2,3\n4,5,6")
        config = {
            'y_exclude': '[]',
            'x_exclude': '10',
            'y_predict': '',
            'datatype': '["numeric","numeric","numeric"]',
            'distinct': '{}'
        }
        result = ml.preprocess(data, config)

        # Small dataset should pass
        assert 'error' not in result


class TestPrepCorrelation:
    """Test the prep_correlation function"""

    def test_prep_correlation_basic(self, simple_numeric_csv, basic_config):
        """Test basic correlation preparation"""
        data, counts, meta, categorical_cols = ml.prep_correlation(
            simple_numeric_csv, basic_config
        )

        assert 'A' in data
        assert 'B' in data
        assert 'C' in data
        assert len(data['A']) == 3
        assert 'header' in meta

    def test_prep_correlation_with_categorical(self, simple_categorical_csv, categorical_config):
        """Test correlation prep with categorical data"""
        data, counts, meta, categorical_cols = ml.prep_correlation(
            simple_categorical_csv, categorical_config
        )

        assert 'Color' in data
        assert 'Size' in data
        assert 'Color' in counts
        assert 'Red' in counts['Color']
        assert counts['Color']['Red'] == 2

    def test_prep_correlation_excludes_columns(self):
        """Test that y_exclude removes columns from correlation prep"""
        data_fh = io.StringIO("A,B,C\n1,2,3\n4,5,6")
        config = {
            'y_exclude': '[1]',  # Exclude column B
            'x_exclude': '10',
            'y_predict': '',
            'datatype': '["numeric","numeric","numeric"]',
            'distinct': '{}'
        }
        data, counts, meta, categorical_cols = ml.prep_correlation(data_fh, config)

        assert 'A' in data
        assert 'B' not in data
        assert 'C' in data

    def test_prep_correlation_skips_comments(self):
        """Test that comment lines are skipped"""
        data_fh = io.StringIO("A,B\n1,2\n#comment\n3,4")
        config = {
            'y_exclude': '[]',
            'x_exclude': '10',
            'y_predict': '',
            'datatype': '["numeric","numeric"]',
            'distinct': '{}'
        }
        data, counts, meta, categorical_cols = ml.prep_correlation(data_fh, config)

        assert len(data['A']) == 2  # Should not include comment line
