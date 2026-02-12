"""
Utility functions for data file handling.
"""


def choose_delimiter(fh):
    """
    Auto-detect delimiter by examining first two lines for consistent tab counts.
    Returns '\t' for TSV, ',' for CSV. Resets file position to start.
    """
    start = (fh.readline(), fh.readline())
    if start[0].count('\t') >= 1 and start[0].count('\t') == start[1].count('\t'):
        delimiter = '\t'
    else:
        delimiter = ','
    fh.seek(0)
    return delimiter
