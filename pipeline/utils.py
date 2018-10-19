import os
import pickle
import importlib
import json
import yaml
import numpy as np
import pandas as pd
import logging

LOGGER = logging.getLogger('luigi-interface')
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def absolute_path_from_project_root(rel_path_from_project_root):
    return os.path.join(PROJECT_ROOT, rel_path_from_project_root)


def create_folder(file_name):
    absolute_dirname = absolute_path_from_project_root(file_name)

    last_element = os.path.basename(absolute_dirname)
    if '.' in last_element: # not a folder
        absolute_dirname = os.path.dirname(absolute_dirname)

    LOGGER.info('Creating ' + absolute_dirname)
    if not os.path.exists(absolute_dirname):
        os.makedirs(absolute_dirname)


def load_data(pathname):
    absolute_pathname = absolute_path_from_project_root(pathname)
    if pathname.endswith(".pickle"):
        with open(absolute_pathname,"rb") as f:
            d = pickle.load(f)
    elif pathname.endswith(".csv"):
        d = pd.read_csv(absolute_pathname)
    elif pathname.endswith(".json"):
        with open(absolute_pathname, 'r') as f:
            d = json.load(f)
    elif pathname.endswith(".txt"):
        with open(absolute_pathname, 'r') as f:
            d = f.read()
    else:
        extension = os.path.basename(pathname).split(".")[-1]
        raise Exception('Unrecognized file extension: "{}"'.format(extension))
    return d


def save_data(data_object, pathname, protocol=2, index=False):
    absolute_pathname = absolute_path_from_project_root(pathname)
    if pathname.endswith(".pickle"):
        with open(absolute_pathname,"wb") as f:
            pickle.dump(data_object,f,protocol=protocol)
    elif pathname.endswith(".csv") and isinstance(data_object, pd.DataFrame):
        data_object.to_csv(absolute_pathname, index=index)
    elif pathname.endswith(".json") and (isinstance(data_object, dict) or (isinstance(data_object, list) and np.all([isinstance(x, dict) for x in data_object]))):
        with open(absolute_pathname, 'w') as f:
            f.write(json.dumps(data_object))
    elif pathname.endswith(".txt") and isinstance(data_object, str):
        with open(absolute_pathname, 'w') as f:
            f.write(data_object)
    else:
        extension = os.path.basename(pathname).split(".")[-1]
        object_type = type(data_object)
        raise Exception('Unrecognized file extension "{}" and type "{}"'.format(extension, object_type))


def import_object(name):
    names = name.split('.')

    if len(names) < 2:
        raise ValueError('invalid object name: ' + str(name) + '. Name should be like "module.object"')

    obj_name = names[-1]

    # qualified name with package and/or modules
    module_name = '.'.join(names[:-1])
    module = importlib.import_module(module_name)

    return getattr(module, obj_name)


def load_yaml_config(filename):
    with open(filename, 'r') as f:
        return yaml.load(f)
