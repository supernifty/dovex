#!/usr/bin/env python
'''
Tests for ml.py prediction algorithms and feature extraction
'''

import io
import ml


class TestEvaluate:
    """Test the evaluate function orchestration"""

    def test_evaluate_returns_error_from_preprocess(self):
        """Test that evaluate propagates preprocessing errors"""
        data = io.StringIO("A\n")  # No data rows
        config = {
            'y_exclude': '[]',
            'x_exclude': '10',
            'y_predict': '',
            'datatype': '["numeric"]',
            'distinct': '{}'
        }
        learner = ml.sklearn.linear_model.LinearRegression()
        result = ml.evaluate(data, config, learner)

        assert 'error' in result

    def test_evaluate_classification_returns_confusion_matrix(self, iris_file_handle, iris_config):
        """Test that classification problems return confusion matrix"""
        learner = ml.sklearn.linear_model.LogisticRegression(solver='lbfgs', multi_class='multinomial')
        result = ml.evaluate(iris_file_handle, iris_config, learner)

        assert 'error' not in result
        assert 'confusion' in result
        assert 'y_labels' in result
        assert len(result['confusion']) > 0

    def test_evaluate_regression_no_confusion_matrix(self):
        """Test that regression problems don't return confusion matrix"""
        data = io.StringIO("A,B,C\n1,2,10\n4,5,20\n7,8,30\n2,3,15\n5,6,25")
        config = {
            'y_exclude': '[]',
            'x_exclude': '10',
            'y_predict': '2',  # Predict numeric column C
            'datatype': '["numeric","numeric","numeric"]',
            'distinct': '{}'
        }
        learner = ml.sklearn.linear_model.LinearRegression()
        result = ml.evaluate(data, config, learner)

        assert 'error' not in result
        assert 'confusion' not in result
        assert 'training_score' in result
        assert 'cross_validation_score' in result

    def test_evaluate_with_feature_extraction(self, iris_file_handle, iris_config):
        """Test that evaluate extracts feature importances when provided"""
        learner = ml.sklearn.ensemble.RandomForestClassifier()
        result = ml.evaluate(iris_file_handle, iris_config, learner, ml.random_forest_features)

        assert 'error' not in result
        assert 'features' in result
        # Iris has 5 columns, target is last, so 4 feature columns
        # map_to_original_features returns one entry per non-target column processed
        assert len(result['features']) == 4  # 4 feature columns


class TestLogisticRegression:
    """Test logistic regression algorithm"""

    def test_logistic_regression_on_iris(self, iris_file_handle, iris_config):
        """Integration: Full logistic regression pipeline on iris"""
        result = ml.logistic_regression(iris_file_handle, iris_config)

        assert 'error' not in result
        assert result['training_score'] > 0.8
        assert result['cross_validation_score'] > 0.8
        assert 'confusion' in result
        assert 'predictions' in result
        assert len(result['predictions']) == 150

    def test_logistic_regression_with_class_weight(self, iris_file_handle, iris_config):
        """Test logistic regression with balanced class weights"""
        iris_config['class_weight'] = 'balanced'
        result = ml.logistic_regression(iris_file_handle, iris_config)

        assert 'error' not in result
        assert 'training_score' in result

    def test_logistic_regression_features_shape(self):
        """Test that feature extractor returns correct shape"""
        learner = ml.sklearn.linear_model.LogisticRegression(solver='lbfgs')
        # Train on simple data
        learner.fit([[1, 2], [3, 4], [5, 6]], ['A', 'B', 'A'])
        features = ml.logistic_regression_features(learner)

        assert len(features) == 2  # 2 input features


class TestSVC:
    """Test Support Vector Classification"""

    def test_svc_on_iris(self, iris_file_handle, iris_config):
        """Integration: Full SVC pipeline on iris"""
        result = ml.svc(iris_file_handle, iris_config)

        assert 'error' not in result
        assert result['training_score'] > 0.8
        assert 'confusion' in result
        assert 'predictions' in result

    def test_svc_features_shape(self):
        """Test that SVC feature extractor returns correct shape"""
        learner = ml.sklearn.svm.LinearSVC()
        learner.fit([[1, 2], [3, 4], [5, 6], [7, 8]], ['A', 'B', 'A', 'B'])
        features = ml.svc_features(learner)

        assert len(features) == 2  # 2 input features


class TestRandomForest:
    """Test Random Forest classifier"""

    def test_random_forest_on_iris(self, iris_file_handle, iris_config):
        """Integration: Full Random Forest pipeline on iris"""
        result = ml.random_forest(iris_file_handle, iris_config)

        assert 'error' not in result
        assert result['training_score'] > 0.8
        assert 'confusion' in result
        assert 'features' in result

    def test_random_forest_features_shape(self):
        """Test that RF feature extractor returns correct shape"""
        learner = ml.sklearn.ensemble.RandomForestClassifier(n_estimators=10, random_state=42)
        learner.fit([[1, 2], [3, 4], [5, 6], [7, 8]], ['A', 'B', 'A', 'B'])
        features = ml.random_forest_features(learner)

        assert len(features) == 2
        assert all(f >= 0 for f in features)  # Feature importances are non-negative


class TestLinearRegression:
    """Test linear regression algorithm"""

    def test_linear_regression_basic(self):
        """Test linear regression on simple data"""
        data = io.StringIO("A,B,C\n1,2,10\n2,3,15\n3,4,20\n4,5,25\n5,6,30")
        config = {
            'y_exclude': '[]',
            'x_exclude': '10',
            'y_predict': '2',
            'datatype': '["numeric","numeric","numeric"]',
            'distinct': '{}'
        }
        result = ml.linear_regression(data, config)

        assert 'error' not in result
        assert 'training_score' in result
        assert 'cross_validation_score' in result
        assert 'confusion' not in result  # No confusion matrix for regression

    def test_linear_regression_features_shape(self):
        """Test that linear regression feature extractor returns correct shape"""
        learner = ml.sklearn.linear_model.LinearRegression()
        learner.fit([[1, 2], [3, 4], [5, 6]], [10, 20, 30])
        features = ml.linear_regression_features(learner)

        assert len(features) == 2


class TestSVR:
    """Test Support Vector Regression"""

    def test_svr_basic(self):
        """Test SVR on simple data"""
        data = io.StringIO("A,B,C\n1,2,10\n2,3,15\n3,4,20\n4,5,25\n5,6,30\n6,7,35")
        config = {
            'y_exclude': '[]',
            'x_exclude': '10',
            'y_predict': '2',
            'datatype': '["numeric","numeric","numeric"]',
            'distinct': '{}'
        }
        result = ml.svr(data, config)

        assert 'error' not in result
        assert 'training_score' in result
        assert 'predictions' in result

    def test_svr_features_shape(self):
        """Test that SVR feature extractor returns correct shape"""
        learner = ml.sklearn.svm.LinearSVR(random_state=42, max_iter=2000)
        learner.fit([[1, 2], [3, 4], [5, 6]], [10, 20, 30])
        features = ml.svr_features(learner)

        assert len(features) == 2


class TestFeatureMapping:
    """Test map_to_original_features helper"""

    def test_map_to_original_features_numeric_only(self):
        """Test feature mapping with only numeric columns"""
        importances = [0.1, 0.2, 0.3]
        y_exclude = set()
        y_predict = None
        distinct = {}
        categorical_cols = set()

        result = ml.map_to_original_features(importances, y_exclude, y_predict, distinct, categorical_cols)

        assert result == [0.1, 0.2, 0.3]

    def test_map_to_original_features_with_exclusion(self):
        """Test feature mapping with excluded columns"""
        importances = [0.1, 0.2]  # 2 features after exclusion
        y_exclude = {1}  # Column 1 excluded
        y_predict = None
        distinct = {}
        categorical_cols = set()

        result = ml.map_to_original_features(importances, y_exclude, y_predict, distinct, categorical_cols)

        assert len(result) == 3  # Original 3 columns
        assert result[0] == 0.1
        assert result[1] == 0  # Excluded column gets 0
        assert result[2] == 0.2

    def test_map_to_original_features_with_categorical(self):
        """Test feature mapping with one-hot encoded categorical"""
        # If column 1 is categorical with 3 distinct values, it becomes 3 features
        importances = [0.1, 0.2, 0.3, 0.4, 0.5]  # col0 (numeric), col1 (3 one-hot), col2 (numeric)
        y_exclude = set()
        y_predict = None
        distinct = {1: 3}
        categorical_cols = {1}

        result = ml.map_to_original_features(importances, y_exclude, y_predict, distinct, categorical_cols)

        assert len(result) == 3  # 3 original columns
        assert result[0] == 0.1
        assert result[1] == 0.2 + 0.3 + 0.4  # Sum of one-hot encoded features
        assert result[2] == 0.5

    def test_map_to_original_features_with_target(self):
        """Test feature mapping with prediction target column"""
        importances = [0.1, 0.2]
        y_exclude = set()
        y_predict = 1  # Column 1 is target
        distinct = {}
        categorical_cols = set()

        result = ml.map_to_original_features(importances, y_exclude, y_predict, distinct, categorical_cols)

        assert len(result) == 3
        assert result[0] == 0.1
        assert result[1] == 0  # Target column gets 0
        assert result[2] == 0.2
