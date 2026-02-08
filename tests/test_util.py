#!/usr/bin/env python
'''
Tests for util.py
'''

import io
import util


class TestChooseDelimiter:
    """Test delimiter detection"""

    def test_detects_comma_delimiter(self):
        """Should detect CSV with comma delimiter"""
        data = io.StringIO("A,B,C\n1,2,3\n4,5,6")
        delimiter = util.choose_delimiter(data)
        assert delimiter == ','
        assert data.tell() == 0  # Should reset file position

    def test_detects_tab_delimiter(self):
        """Should detect TSV with tab delimiter"""
        data = io.StringIO("A\tB\tC\n1\t2\t3\n4\t5\t6")
        delimiter = util.choose_delimiter(data)
        assert delimiter == '\t'
        assert data.tell() == 0

    def test_defaults_to_comma_with_no_tabs(self):
        """Should default to comma when no consistent tabs"""
        data = io.StringIO("A,B,C\n1,2,3")
        delimiter = util.choose_delimiter(data)
        assert delimiter == ','

    def test_requires_consistent_tabs_in_both_lines(self):
        """Should require same number of tabs in first two lines"""
        data = io.StringIO("A\tB\tC\n1,2,3")  # Tab in first, comma in second
        delimiter = util.choose_delimiter(data)
        assert delimiter == ','

    def test_single_tab_is_sufficient(self):
        """Should detect tab with just one tab per line"""
        data = io.StringIO("A\tB\n1\t2")
        delimiter = util.choose_delimiter(data)
        assert delimiter == '\t'

    def test_handles_empty_lines(self):
        """Should handle edge cases gracefully"""
        data = io.StringIO("A,B\n")
        delimiter = util.choose_delimiter(data)
        assert delimiter == ','
