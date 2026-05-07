
from errno import EEXIST
from glob import glob
from os import makedirs, path
import os
import re


def mkdir_p(folder_path):
    try:
        makedirs(folder_path)
    except OSError as exc: # Python >2.5
        if exc.errno == EEXIST and path.isdir(folder_path):
            pass
        else:
            raise


def extract_trailing_integer(text):
    match = re.search(r"(\d+)(?!.*\d)", str(text))
    if match is None:
        return None
    return int(match.group(1))


def checkpoint_sort_key(checkpoint_path):
    iteration = extract_trailing_integer(os.path.basename(checkpoint_path))
    if iteration is None:
        return (-1, os.path.basename(checkpoint_path))
    return (iteration, os.path.basename(checkpoint_path))


def find_latest_checkpoint(model_path):
    checkpoint_candidates = glob(os.path.join(model_path, "chkpnt*.pth"))
    if not checkpoint_candidates:
        return None
    return sorted(checkpoint_candidates, key=checkpoint_sort_key)[-1]


def searchForMaxIteration(folder):
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Iteration folder does not exist: {folder}")

    saved_iters = []
    for fname in os.listdir(folder):
        iteration = extract_trailing_integer(fname)
        if iteration is not None:
            saved_iters.append(iteration)

    if not saved_iters:
        raise ValueError(f"No saved iterations were found in: {folder}")
    return max(saved_iters)

