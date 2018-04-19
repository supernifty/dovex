
def choose_delimiter(fh):
    start = (fh.readline(), fh.readline())
    if start[0].count('\t') >= 1 and start[0].count('\t') == start[1].count('\t'):
        delimiter = '\t'
    else:
        delimiter = ','
    fh.seek(0)
    return delimiter
