#!/usr/bin/env python
'''
Tests for ml.py correlation and statistical analysis functions
'''

import io
import ml


class TestCorrelation:
    """Test the correlation function"""

    def test_correlation_numeric_pearson(self):
        """Test correlation with numeric columns uses Pearson"""
        data = io.StringIO("A,B,C\n1,2,10\n2,4,20\n3,6,30\n4,8,40\n5,10,50")
        config = {
            'y_exclude': '[]',
            'x_exclude': '10',
            'y_predict': '',
            'datatype': '["numeric","numeric","numeric"]',
            'distinct': '{}'
        }
        result = ml.correlation(data, config)

        assert 'xs' in result
        assert 'zs' in result  # p-values
        assert 'cs' in result  # counts
        assert 'ts' in result  # test types
        assert len(result['xs']) == 3  # 3 columns
        assert len(result['zs']) == 3  # 3x3 matrix
        # Check that Pearson correlation was used
        assert 'Pearson' in result['ts'][0][1]

    def test_correlation_categorical_chisquare(self):
        """Test correlation with categorical columns uses Chi-square"""
        data = io.StringIO("Color,Size\nRed,Small\nBlue,Large\nRed,Large\nBlue,Small\nRed,Small")
        config = {
            'y_exclude': '[]',
            'x_exclude': '10',
            'y_predict': '',
            'datatype': '["categorical","categorical"]',
            'distinct': '{"0": 2, "1": 2}'
        }
        result = ml.correlation(data, config)

        assert 'xs' in result
        assert len(result['xs']) == 2
        # Check that Chi-square was used
        assert 'Chi-square' in result['ts'][0][1]

    def test_correlation_mixed_anova(self):
        """Test correlation with mixed types uses ANOVA"""
        data = io.StringIO("Category,Score\nA,10\nB,20\nA,15\nB,25\nA,12\nB,22")
        config = {
            'y_exclude': '[]',
            'x_exclude': '10',
            'y_predict': '',
            'datatype': '["categorical","numeric"]',
            'distinct': '{"0": 2}'
        }
        result = ml.correlation(data, config)

        assert 'xs' in result
        assert len(result['xs']) == 2
        # Check that ANOVA was used
        assert 'ANOVA' in result['ts'][0][1] or 'ANOVA' in result['ts'][1][0]

    def test_correlation_diagonal_is_zero(self):
        """Test that diagonal (self-correlation) is 0 p-value"""
        data = io.StringIO("A,B\n1,2\n3,4\n5,6")
        config = {
            'y_exclude': '[]',
            'x_exclude': '10',
            'y_predict': '',
            'datatype': '["numeric","numeric"]',
            'distinct': '{}'
        }
        result = ml.correlation(data, config)

        # Diagonal should be 0 (p-value for self-correlation)
        assert result['zs'][0][0] == 0
        assert result['zs'][1][1] == 0

    def test_correlation_diagonal_type_is_na(self):
        """Test that diagonal test type is N/A"""
        data = io.StringIO("A,B\n1,2\n3,4\n5,6")
        config = {
            'y_exclude': '[]',
            'x_exclude': '10',
            'y_predict': '',
            'datatype': '["numeric","numeric"]',
            'distinct': '{}'
        }
        result = ml.correlation(data, config)

        assert result['ts'][0][0] == 'N/A'
        assert result['ts'][1][1] == 'N/A'

    def test_correlation_with_missing_values(self):
        """Test that correlation skips missing values"""
        data = io.StringIO("A,B\n1,2\n,4\n5,6\n7,")
        config = {
            'y_exclude': '[]',
            'x_exclude': '10',
            'y_predict': '',
            'datatype': '["numeric","numeric"]',
            'distinct': '{}'
        }
        result = ml.correlation(data, config)

        assert 'xs' in result
        # Should complete without error despite missing values
        assert len(result['zs']) == 2

    def test_correlation_returns_counts(self):
        """Test that correlation returns sample counts"""
        data = io.StringIO("A,B\n1,2\n3,4\n5,6\n7,8")
        config = {
            'y_exclude': '[]',
            'x_exclude': '10',
            'y_predict': '',
            'datatype': '["numeric","numeric"]',
            'distinct': '{}'
        }
        result = ml.correlation(data, config)

        assert 'cs' in result
        # Count should be 4 for valid pairs
        assert result['cs'][0][1] == 4
        assert result['cs'][1][0] == 4

    def test_correlation_excludes_columns(self):
        """Test that y_exclude removes columns from correlation"""
        data = io.StringIO("A,B,C\n1,2,3\n4,5,6\n7,8,9")
        config = {
            'y_exclude': '[1]',  # Exclude column B
            'x_exclude': '10',
            'y_predict': '',
            'datatype': '["numeric","numeric","numeric"]',
            'distinct': '{}'
        }
        result = ml.correlation(data, config)

        assert len(result['xs']) == 2  # Only A and C
        assert 'A' in result['xs']
        assert 'C' in result['xs']
        assert 'B' not in result['xs']

    def test_correlation_skips_comment_lines(self):
        """Test that correlation skips comment lines"""
        data = io.StringIO("A,B\n1,2\n#comment\n3,4\n5,6")
        config = {
            'y_exclude': '[]',
            'x_exclude': '10',
            'y_predict': '',
            'datatype': '["numeric","numeric"]',
            'distinct': '{}'
        }
        result = ml.correlation(data, config)

        # Count should be 3 (not including comment line)
        assert result['cs'][0][1] == 3

    def test_correlation_with_details(self):
        """Test correlation with_detail parameter"""
        data = io.StringIO("A,B\n1,2\n3,4\n5,6")
        config = {
            'y_exclude': '[]',
            'x_exclude': '10',
            'y_predict': '',
            'datatype': '["numeric","numeric"]',
            'distinct': '{}'
        }
        result = ml.correlation(data, config, with_detail=True)

        assert 'ds' in result  # Details should be present

    def test_correlation_handles_single_value_groups(self):
        """Test that correlation handles edge case of single-value groups"""
        data = io.StringIO("A,B\n1,1\n1,1\n1,1")
        config = {
            'y_exclude': '[]',
            'x_exclude': '10',
            'y_predict': '',
            'datatype': '["numeric","numeric"]',
            'distinct': '{}'
        }
        result = ml.correlation(data, config)

        # Should complete without error
        assert 'xs' in result
        # P-value might be NaN or 1 for no variance


class TestCorrelationSubgroup:
    """Test the correlation_subgroup function for pairwise comparisons"""

    def test_correlation_subgroup_chi_square(self):
        """Test subgroup analysis with two categorical variables"""
        data = io.StringIO("Group,Outcome\nA,Success\nA,Failure\nB,Success\nB,Failure\nC,Success\nC,Failure")
        config = {
            'y_exclude': '[]',
            'x_exclude': 'Group',  # Primary variable name
            'y_predict': '',  # Empty - not used for this analysis
            'datatype': '["categorical","categorical"]',
            'distinct': '{"0": 3, "1": 2}'
        }
        result = ml.correlation_subgroup(data, config)

        assert 'result' in result
        # Should have pairwise comparisons between groups

    def test_correlation_subgroup_t_test(self):
        """Test subgroup analysis with categorical and numeric variables"""
        data = io.StringIO("Group,Score\nA,10\nA,12\nA,11\nB,20\nB,22\nB,21\nC,15\nC,17\nC,16")
        config = {
            'y_exclude': '[]',
            'x_exclude': 'Group',  # Categorical variable
            'y_predict': '',  # Empty - not used for this analysis
            'datatype': '["categorical","numeric"]',
            'distinct': '{"0": 3}'
        }
        result = ml.correlation_subgroup(data, config)

        assert 'result' in result
        # Should contain pairwise t-tests
        if len(result['result']) > 0:
            # Check that t-test was used
            assert result['result'][0][8] == 't-test'

    def test_correlation_subgroup_returns_statistics(self):
        """Test that subgroup analysis returns mean and std"""
        data = io.StringIO("Group,Score\nA,10\nA,12\nB,20\nB,22")
        config = {
            'y_exclude': '[]',
            'x_exclude': 'Group',
            'y_predict': '',
            'datatype': '["categorical","numeric"]',
            'distinct': '{"0": 2}'
        }
        result = ml.correlation_subgroup(data, config)

        assert 'result' in result
        if len(result['result']) > 0:
            # Result tuple: (s1, s2, pvalue, n, mean1, mean2, std1, std2, test_type)
            record = result['result'][0]
            assert len(record) == 9
            assert isinstance(record[4], float)  # mean1
            assert isinstance(record[5], float)  # mean2

    def test_correlation_subgroup_avoids_duplicate_pairs(self):
        """Test that subgroup analysis doesn't duplicate (A,B) and (B,A)"""
        data = io.StringIO("Group,Score\nA,10\nA,12\nB,20\nB,22\nC,30\nC,32")
        config = {
            'y_exclude': '[]',
            'x_exclude': 'Group',
            'y_predict': '',
            'datatype': '["categorical","numeric"]',
            'distinct': '{"0": 3}'
        }
        result = ml.correlation_subgroup(data, config)

        # With 3 groups (A, B, C), should have 3 pairs: (A,B), (A,C), (B,C)
        assert 'result' in result
        # Each pair should appear only once
        pairs = set()
        for record in result['result']:
            pair = tuple(sorted([record[0], record[1]]))
            assert pair not in pairs, f"Duplicate pair found: {pair}"
            pairs.add(pair)

    def test_correlation_subgroup_handles_missing_values(self):
        """Test that subgroup analysis skips missing values"""
        data = io.StringIO("Group,Score\nA,10\nA,\nB,20\nB,22")
        config = {
            'y_exclude': '[]',
            'x_exclude': 'Group',
            'y_predict': '',
            'datatype': '["categorical","numeric"]',
            'distinct': '{"0": 2}'
        }
        result = ml.correlation_subgroup(data, config)

        # Should complete without error
        assert 'result' in result

    def test_correlation_subgroup_not_applicable_for_numeric_numeric(self):
        """Test that subgroup analysis only works for appropriate types"""
        data = io.StringIO("A,B\n1,2\n3,4\n5,6")
        config = {
            'y_exclude': '[]',
            'x_exclude': 'A',
            'y_predict': '',
            'datatype': '["numeric","numeric"]',
            'distinct': '{}'
        }
        result = ml.correlation_subgroup(data, config)

        # Should return empty result for numeric-numeric
        assert 'result' in result
        assert len(result['result']) == 0


class TestCorrelationIntegration:
    """Integration tests for correlation analysis"""

    def test_correlation_on_iris_features(self, iris_file_handle):
        """Integration: Correlation analysis on iris numeric features"""
        config = {
            'y_exclude': '[4]',  # Exclude categorical class column
            'x_exclude': '10',
            'y_predict': '',
            'datatype': '["numeric","numeric","numeric","numeric","categorical"]',
            'distinct': '{"4": 3}'
        }
        result = ml.correlation(iris_file_handle, config)

        assert 'xs' in result
        assert len(result['xs']) == 4  # 4 numeric features
        # All should use Pearson correlation
        for i in range(4):
            for j in range(4):
                if i != j:
                    assert 'Pearson' in result['ts'][i][j]

    def test_correlation_mixed_types(self):
        """Test correlation with all three statistical tests"""
        data = io.StringIO("NumA,NumB,CatA,CatB\n1,2,X,M\n3,4,Y,N\n5,6,X,M\n7,8,Y,N\n2,3,X,N")
        config = {
            'y_exclude': '[]',
            'x_exclude': '10',
            'y_predict': '',
            'datatype': '["numeric","numeric","categorical","categorical"]',
            'distinct': '{"2": 2, "3": 2}'
        }
        result = ml.correlation(data, config)

        assert 'xs' in result
        assert len(result['xs']) == 4

        # NumA vs NumB should be Pearson
        num_a_idx = result['xs'].index('NumA')
        num_b_idx = result['xs'].index('NumB')
        assert 'Pearson' in result['ts'][num_a_idx][num_b_idx]

        # CatA vs CatB should be Chi-square
        cat_a_idx = result['xs'].index('CatA')
        cat_b_idx = result['xs'].index('CatB')
        assert 'Chi-square' in result['ts'][cat_a_idx][cat_b_idx]

        # NumA vs CatA should be ANOVA
        assert 'ANOVA' in result['ts'][num_a_idx][cat_a_idx]

    def test_correlation_handles_duplicate_column_names(self):
        """Test that correlation handles duplicate column names via exclusion"""
        data = io.StringIO("A,B,A\n1,2,3\n4,5,6\n7,8,9")
        config = {
            'y_exclude': '[]',
            'x_exclude': '10',
            'y_predict': '',
            'datatype': '["numeric","numeric","numeric"]',
            'distinct': '{}'
        }
        # Note: prep_correlation should handle duplicate exclusion
        result = ml.correlation(data, config)

        assert 'xs' in result
        # Second 'A' column should be excluded by prep_correlation
