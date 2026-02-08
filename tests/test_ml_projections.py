#!/usr/bin/env python
'''
Tests for ml.py dimensionality reduction and projection functions
'''

import io
import ml


class TestProject:
    """Test the project function orchestration"""

    def test_project_returns_error_from_preprocess(self):
        """Test that project propagates preprocessing errors"""
        data = io.StringIO("A\n")  # No data rows
        config = {
            'y_exclude': '[]',
            'x_exclude': '10',
            'y_predict': '',
            'datatype': '["numeric"]',
            'distinct': '{}'
        }
        projector = ml.sklearn.decomposition.PCA(n_components=2)
        result = ml.project(data, config, projector, has_features=True)

        assert 'error' in result

    def test_project_with_max_rows_limit(self):
        """Test that project enforces max_rows limit"""
        # Create data with 11 rows (exceeds limit of 10)
        data_lines = ["A,B,C\n"] + [f"{i},{i+1},{i+2}\n" for i in range(11)]
        data = io.StringIO("".join(data_lines))
        config = {
            'y_exclude': '[]',
            'x_exclude': '10',
            'y_predict': '',
            'datatype': '["numeric","numeric","numeric"]',
            'distinct': '{}'
        }
        projector = ml.sklearn.decomposition.PCA(n_components=2)
        result = ml.project(data, config, projector, has_features=True, max_rows=10)

        assert 'error' in result
        assert 'Too many rows' in result['error']

    def test_project_with_features_returns_feature_importance(self, iris_file_handle, iris_config):
        """Test that project with has_features=True returns feature weights"""
        projector = ml.sklearn.decomposition.PCA(n_components=2)
        result = ml.project(iris_file_handle, iris_config, projector, has_features=True)

        assert 'error' not in result
        assert 'projection' in result
        assert 'features' in result
        assert 'features_2' in result  # Second component features

    def test_project_without_features_no_feature_importance(self):
        """Test that project with has_features=False doesn't return features"""
        data = io.StringIO("A,B,C\n1,2,3\n4,5,6\n7,8,9\n10,11,12\n13,14,15")
        config = {
            'y_exclude': '[]',
            'x_exclude': '10',
            'y_predict': '',
            'datatype': '["numeric","numeric","numeric"]',
            'distinct': '{}'
        }
        projector = ml.sklearn.manifold.MDS(n_components=2, max_iter=100)
        result = ml.project(data, config, projector, has_features=False)

        assert 'error' not in result
        assert 'projection' in result
        assert 'features' not in result
        assert 'features_2' not in result


class TestPCA:
    """Test Principal Component Analysis"""

    def test_pca_on_iris(self, iris_file_handle, iris_config):
        """Integration: Full PCA pipeline on iris"""
        result = ml.pca(iris_file_handle, iris_config)

        assert 'error' not in result
        assert 'projection' in result
        assert len(result['projection']) == 150  # 150 iris samples
        assert len(result['projection'][0]) == 2  # 2D projection
        assert 'features' in result
        assert 'features_2' in result

    def test_pca_basic(self):
        """Test PCA on simple numeric data"""
        data = io.StringIO("A,B,C\n1,2,3\n4,5,6\n7,8,9\n2,3,4\n5,6,7")
        config = {
            'y_exclude': '[]',
            'x_exclude': '10',
            'y_predict': '',
            'datatype': '["numeric","numeric","numeric"]',
            'distinct': '{}'
        }
        result = ml.pca(data, config)

        assert 'error' not in result
        assert 'projection' in result
        assert len(result['projection']) == 5  # 5 data points
        assert len(result['projection'][0]) == 2  # 2 components

    def test_pca_no_max_rows_limit(self):
        """Test that PCA has no max_rows limit"""
        # Create large dataset (>1000 rows, which would fail MDS)
        data_lines = ["A,B\n"] + [f"{i},{i+1}\n" for i in range(1500)]
        data = io.StringIO("".join(data_lines))
        config = {
            'y_exclude': '[]',
            'x_exclude': '10',
            'y_predict': '',
            'datatype': '["numeric","numeric"]',
            'distinct': '{}'
        }
        result = ml.pca(data, config)

        assert 'error' not in result
        assert len(result['projection']) == 1500


class TestMDS:
    """Test Multidimensional Scaling"""

    def test_mds_basic(self):
        """Test MDS on small dataset"""
        data = io.StringIO("A,B,C\n1,2,3\n4,5,6\n7,8,9\n2,3,4\n5,6,7")
        config = {
            'y_exclude': '[]',
            'x_exclude': '10',
            'y_predict': '',
            'datatype': '["numeric","numeric","numeric"]',
            'distinct': '{}'
        }
        result = ml.mds(data, config)

        assert 'error' not in result
        assert 'projection' in result
        assert len(result['projection']) == 5
        assert 'features' not in result  # MDS doesn't have features

    def test_mds_enforces_max_rows_limit(self):
        """Test that MDS enforces MAX_ROWS['mds'] limit (1000)"""
        # Create dataset with 1001 rows
        data_lines = ["A,B\n"] + [f"{i},{i+1}\n" for i in range(1001)]
        data = io.StringIO("".join(data_lines))
        config = {
            'y_exclude': '[]',
            'x_exclude': '10',
            'y_predict': '',
            'datatype': '["numeric","numeric"]',
            'distinct': '{}'
        }
        result = ml.mds(data, config)

        assert 'error' in result
        assert 'Too many rows' in result['error']
        assert '1001' in result['error']
        assert '1000' in result['error']

    def test_mds_allows_exactly_max_rows(self):
        """Test that MDS allows exactly MAX_ROWS['mds'] rows"""
        # Create dataset with exactly 1000 rows
        data_lines = ["A,B\n"] + [f"{i},{i+1}\n" for i in range(1000)]
        data = io.StringIO("".join(data_lines))
        config = {
            'y_exclude': '[]',
            'x_exclude': '10',
            'y_predict': '',
            'datatype': '["numeric","numeric"]',
            'distinct': '{}'
        }
        result = ml.mds(data, config)

        assert 'error' not in result
        assert len(result['projection']) == 1000


class TestTSNE:
    """Test t-SNE dimensionality reduction"""

    def test_tsne_basic(self):
        """Test t-SNE on small dataset"""
        data = io.StringIO("A,B,C\n1,2,3\n4,5,6\n7,8,9\n2,3,4\n5,6,7\n3,4,5\n6,7,8")
        config = {
            'y_exclude': '[]',
            'x_exclude': '10',
            'y_predict': '',
            'datatype': '["numeric","numeric","numeric"]',
            'distinct': '{}',
            'perplexity': '2'  # Low perplexity for small dataset
        }
        result = ml.tsne(data, config)

        assert 'error' not in result
        assert 'projection' in result
        assert len(result['projection']) == 7
        assert 'features' not in result  # t-SNE doesn't have features

    def test_tsne_enforces_max_rows_limit(self):
        """Test that t-SNE enforces MAX_ROWS['tsne'] limit (10000)"""
        # Create dataset with 10001 rows
        data_lines = ["A,B\n"] + [f"{i},{i+1}\n" for i in range(10001)]
        data = io.StringIO("".join(data_lines))
        config = {
            'y_exclude': '[]',
            'x_exclude': '10',
            'y_predict': '',
            'datatype': '["numeric","numeric"]',
            'distinct': '{}',
            'perplexity': '30'
        }
        result = ml.tsne(data, config)

        assert 'error' in result
        assert 'Too many rows' in result['error']

    def test_tsne_respects_perplexity_config(self):
        """Test that t-SNE uses perplexity from config"""
        data = io.StringIO("A,B,C\n1,2,3\n4,5,6\n7,8,9\n2,3,4\n5,6,7\n3,4,5\n6,7,8\n8,9,10\n1,3,5")
        config = {
            'y_exclude': '[]',
            'x_exclude': '10',
            'y_predict': '',
            'datatype': '["numeric","numeric","numeric"]',
            'distinct': '{}',
            'perplexity': '3'
        }
        result = ml.tsne(data, config)

        assert 'error' not in result
        assert 'projection' in result


class TestProjectionFeatures:
    """Test projection_features helper"""

    def test_projection_features_first_component(self):
        """Test that projection_features extracts first component by default"""
        projector = ml.sklearn.decomposition.PCA(n_components=2)
        # Simple data
        X = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [2, 3, 4]]
        projector.fit(X)

        features = ml.projection_features(projector, component=0)

        assert len(features) == 3  # 3 input features
        assert all(isinstance(f, float) for f in features)
        assert all(f >= 0 for f in features)  # Should be absolute values

    def test_projection_features_second_component(self):
        """Test that projection_features can extract second component"""
        projector = ml.sklearn.decomposition.PCA(n_components=2)
        X = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [2, 3, 4]]
        projector.fit(X)

        features_0 = ml.projection_features(projector, component=0)
        features_1 = ml.projection_features(projector, component=1)

        assert len(features_0) == len(features_1)
        # Features from different components should generally differ
        # (Could be same by coincidence but unlikely with real data)

    def test_projection_features_uses_absolute_values(self):
        """Test that projection_features returns absolute values"""
        projector = ml.sklearn.decomposition.PCA(n_components=2)
        X = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [2, 3, 4]]
        projector.fit(X)

        features = ml.projection_features(projector, component=0)

        # All values should be non-negative (absolute values)
        assert all(f >= 0 for f in features)


class TestProjectionIntegration:
    """Integration tests for projection pipeline"""

    def test_pca_with_categorical_columns(self):
        """Test PCA with mixed numeric/categorical data"""
        data = io.StringIO("Age,Score,Category\n25,100,A\n30,85,B\n35,90,A\n40,95,B\n28,88,A")
        config = {
            'y_exclude': '[]',
            'x_exclude': '10',
            'y_predict': '',
            'datatype': '["numeric","numeric","categorical"]',
            'distinct': '{"2": 2}'
        }
        result = ml.pca(data, config)

        assert 'error' not in result
        assert 'projection' in result
        # Should have 2 numeric + 2 one-hot = 4 total features
        assert len(result['features']) == 3  # Original column count

    def test_pca_with_excluded_columns(self):
        """Test PCA with some columns excluded"""
        data = io.StringIO("A,B,C,D\n1,2,3,4\n5,6,7,8\n9,10,11,12\n2,3,4,5\n6,7,8,9")
        config = {
            'y_exclude': '[1]',  # Exclude column B
            'x_exclude': '10',
            'y_predict': '',
            'datatype': '["numeric","numeric","numeric","numeric"]',
            'distinct': '{}'
        }
        result = ml.pca(data, config)

        assert 'error' not in result
        assert 'projection' in result
        # Features should have 4 entries (one per original column), with 0 at excluded position
        assert len(result['features']) == 4
        assert result['features'][1] == 0  # Excluded column B has 0 importance

    def test_projection_includes_imputation_notes(self):
        """Test that projections include preprocessing notes"""
        data = io.StringIO("A,B,C\n1,2,3\n4,,6\n7,8,9\n2,3,4")
        config = {
            'y_exclude': '[]',
            'x_exclude': '10',
            'y_predict': '',
            'datatype': '["numeric","numeric","numeric"]',
            'distinct': '{}'
        }
        result = ml.pca(data, config)

        assert 'error' not in result
        assert 'notes' in result
        assert len(result['notes']) > 0
        assert 'imputed' in result['notes'][0].lower()
