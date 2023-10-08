"""
using pickle to upload and load data
"""
import pickle
import zipfile
from config import GRAPH_CATEGORIES

def save_meta_data_to_file(path, data):
    """
    save meta data to file
    :return:
    """

    with open(path, 'wb') as f:
        data = pickle.dump(data, f,3)

def from_yaml_to_python(path):
    """
    fro meta data file to python dict
    :return:
    """
    with open(path, "rb") as f:
        data = pickle.load(f)
        # loading file
        return (data)

def read_zip(path):
    with zipfile.ZipFile(path, 'r') as zip_ref:
        data = pickle.loads(zip_ref.read(GRAPH_CATEGORIES.replace(".zip","")))
    return data