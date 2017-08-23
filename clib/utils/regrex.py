import re


def is_path(path):
    regex = r'(.|..)?/.*'
    pattern = re.compile(regex)
    if re.match(regex, path):
        return True
    else:
        return False
