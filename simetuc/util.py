# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 16:45:32 2016

@author: Pedro
"""
import tempfile
from contextlib import contextmanager
import os


# http://stackoverflow.com/a/11892712
@contextmanager
def temp_config_filename(data):
    '''Creates a temporary file and writes text data to it. It returns its filename.
        It deletes the file after use in a context manager.
    '''
    # file won't be deleted after closing
    temp = tempfile.NamedTemporaryFile(mode='wt', delete=False)
    if data:
        temp.write(data)
    temp.close()
    yield temp.name
    os.unlink(temp.name)  # delete file


@contextmanager
def temp_bin_filename():
    '''Creates a temporary binary file. It returns its filename.
        It deletes the file after use in a context manager.
    '''
    # file won't be deleted after closing
    temp = tempfile.NamedTemporaryFile(mode='wb', delete=False)
    temp.close()
    yield temp.name
    os.unlink(temp.name)  # delete file
