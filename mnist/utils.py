import os


def file_choices(choices, fname):
    ext = os.path.splitext(fname)[1][1:]
    if ext not in choices:
        raise (NameError("file doesn't end with {}".format(choices)))
    return fname