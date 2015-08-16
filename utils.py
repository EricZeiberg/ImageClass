import os

"""
This function is the same as os.listdir() but ignores hidden files
"""
def listdir_hidden(path):
    for f in os.listdir(path):
        if not f.startswith('.'):
            yield f
