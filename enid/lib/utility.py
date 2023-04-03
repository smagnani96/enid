# Copyright 2023 ENID
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utility file containing all the function used among all the programs and modules."""
import datetime
import importlib
import inspect
import json
import logging
import math
import os
import pkgutil
import re
import shutil
import sys
import typing
from ctypes import Array, Structure
from ctypes import Union as CUnion
from ctypes import _Pointer, _SimpleCData
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from json import JSONEncoder
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import psutil
from pypacker.layer12.ethernet import Ethernet


def set_seed():
    "Function for resetting the seed if "
    seed = os.environ.get("PYTHONHASHSEED", None)

    if seed is not None:
        import tensorflow as tf
        import random

        seed = int(seed)
        random.seed(seed)
        np.random.seed(seed)
        tf.keras.utils.set_random_seed(seed)


def silence():
    import warnings
    import tensorflow as tf
    import tensorflow.python.util.deprecation as deprecation
    from sklearn.exceptions import UndefinedMetricWarning

    """Function for silencing warning and tensorflow logging"""
    warnings.filterwarnings(action='ignore', category=UndefinedMetricWarning)
    warnings.filterwarnings(action='ignore', category=RuntimeWarning)
    warnings.filterwarnings(action='ignore', category=UserWarning)
    np.seterr(all="ignore")
    deprecation._PRINT_DEPRECATION_WARNINGS = False
    tf.get_logger().setLevel('ERROR')
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


class EthInPcap(Ethernet):
    """Personalised class to insert a reference to the dataset, category and pcap"""

    def __init__(self, timestamp: int, dataset: str, category: str, pcap: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.timestamp = timestamp
        self.dataset = dataset
        self.category = category
        self.pcap = pcap


class CDataJSONEncoder(JSONEncoder):
    """A JSON Encoder that puts small containers on single lines."""

    CONTAINER_TYPES = (list, tuple, dict)
    """Container datatypes include primitives or other containers."""

    MAX_WIDTH = 120
    """Maximum width of a container that might be put on a single line."""

    MAX_ITEMS = 15
    """Maximum number of items in container that might be put on single line."""

    def __init__(self, *args, **kwargs):
        # using this class without indentation is pointless
        if kwargs.get("indent") is None:
            kwargs.update({"indent": 1})
        super().__init__(*args, **kwargs)
        self.indentation_level = 0

    def encode(self, o):
        """Encode JSON object *o* with respect to single line lists."""
        o = self.default(o)

        if isinstance(o, (list, tuple)):
            if self._put_on_single_line(o):
                return "[" + ", ".join(self.encode(el) for el in o) + "]"
            else:
                self.indentation_level += 1
                output = [self.indent_str + self.encode(el) for el in o]
                self.indentation_level -= 1
                return "[\n" + ",\n".join(output) + "\n" + self.indent_str + "]"
        elif isinstance(o, dict):
            if o:
                if self._put_on_single_line(o):
                    return "{ " + ", ".join(f"{self.encode(k)}: {self.encode(el)}" for k, el in o.items()) + " }"
                else:
                    self.indentation_level += 1
                    output = [
                        self.indent_str + f"{json.dumps(k)}: {self.encode(v)}" for k, v in o.items()]
                    self.indentation_level -= 1
                    return "{\n" + ",\n".join(output) + "\n" + self.indent_str + "}"
            else:
                return "{}"
        elif isinstance(o, str):  # escape newlines
            o = o.replace("\n", "\\n")
            return f'"{o}"'
        else:
            return json.dumps(o)

    def iterencode(self, o, **kwargs):
        """Required to also work with `json.dump`."""
        return self.encode(o)

    def _put_on_single_line(self, o):
        return self._primitives_only(o) and len(o) <= self.MAX_ITEMS and len(str(o)) - 2 <= self.MAX_WIDTH

    def _primitives_only(self, o: typing.Union[list, tuple, dict]):
        if isinstance(o, (list, tuple)):
            return not any(isinstance(el, self.CONTAINER_TYPES) for el in o)
        elif isinstance(o, dict):
            return not any(isinstance(el, self.CONTAINER_TYPES) for el in o.values())

    @property
    def indent_str(self) -> str:
        if isinstance(self.indent, int):
            return " " * (self.indentation_level * self.indent)
        elif isinstance(self.indent, str):
            return self.indentation_level * self.indent
        else:
            raise ValueError(
                f"indent must either be of type int or str (is: {type(self.indent)})")

    def default(self, obj):
        if inspect.isclass(obj):
            return obj.__name__

        if isinstance(obj, (Array, list)):
            return [self.default(e) for e in obj]

        if isinstance(obj, _Pointer):
            return self.default(obj.contents) if obj else None

        if isinstance(obj, _SimpleCData):
            return self.default(obj.value)

        if isinstance(obj, (bool, int, float, str)):
            return obj

        if obj is None:
            return obj

        if isinstance(obj, Enum):
            return obj.value

        if isinstance(obj, (Structure, CUnion)):
            result = {}
            anonymous = getattr(obj, '_anonymous_', [])

            for key, *_ in getattr(obj, '_fields_', []):
                value = getattr(obj, key)

                # private fields don't encode
                if key.startswith('_'):
                    continue

                if key in anonymous:
                    result.update(self.default(value))
                else:
                    result[key] = self.default(value)

            return result

        if is_dataclass(obj):
            if hasattr(obj, "to_json"):
                return obj.to_json()
            else:
                return {k.name: self.default(getattr(obj, k.name)) for k in fields(obj)}

        if isinstance(obj, dict):
            if obj and not isinstance(next(iter(obj), None), (int, float, str, bool)):
                return [{'key': self.default(k), 'value': self.default(v)} for k, v in obj.items()]
            else:
                return {k: self.default(v) for k, v in obj.items()}

        if isinstance(obj, tuple):
            if hasattr(obj, "_asdict"):
                return self.default(obj._asdict())
            else:
                return [self.default(e) for e in obj]

        if isinstance(obj, np.ndarray):
            return obj.tolist()

        if isinstance(obj, np.integer):
            return int(obj)

        if isinstance(obj, np.floating):
            return float(obj)

        return JSONEncoder.default(self, obj)


def get_best_comb(n_comb):
    """Function to get the best n° of combinations during a randomised
    search of hyperparameters"""
    return n_comb if n_comb < 50 else 50


def get_best_dispatch(size: int):
    """Function to get the best dispatch n° of workers while doing
    a randomised search of hyperparameters"""
    total = psutil.virtual_memory().total
    n_times = 0

    while True:
        if size * n_times > total:
            break
        n_times += 1
    return min(math.floor(math.sqrt(n_times)), os.cpu_count())*2 or 1


def handler(x, y):
    """Utility to call function x with all parameters inside the arg y"""
    return x(*y)


def sort_keep_comb(x, y):
    """Sort array x and y while preserving their labels and returning
    the permutations used"""
    p = np.random.permutation(len(x))
    return x[p], y[p], p


def create_dir(name, overwrite=None):
    """Function to create a directory and, in case, overwrite or
    backup the old one if already present"""
    try:
        os.makedirs(name)
    except FileExistsError:
        if overwrite is False:
            shutil.move(
                name, f"{name}_{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}______backup")
            os.makedirs(name)
        elif overwrite is True:
            shutil.rmtree(name)
            os.makedirs(name)


def load_json_data(path: str, fail=True):
    """Function to load json file into a dictionary"""
    if os.path.isfile(path):
        with open(path, "r") as fp:
            return json.load(fp)
    elif fail:
        raise FileNotFoundError("File {} not found".format(path))
    else:
        return {}


def dump_json_data(data, path: str = None):
    """Function to dump dictionary or dataclass into json file
    with the custom encoder"""
    if path is None:
        return json.dumps(data, cls=CDataJSONEncoder)
    with open(path, 'w') as fp:
        json.dump(data, fp, indent=2, cls=CDataJSONEncoder)


def snake_to_camel(name, join_char=''):
    """Function to transform a string from snake to camel case"""
    return join_char.join(word.title() for word in name.split('_'))


def camel_to_snake(name):
    """Function to transform a string from camel case to snake"""
    return re.sub(r'(?<!^)(?=[A-Z])', '_', name).lower().replace(" ", "")


def safe_division(a, b, default=None):
    """Function to perform a safe division, meaning no exceptions are thrown
    in case of a division by 0 or infinite number"""
    ret = np.divide(a, b)
    if default is not None:
        return np.nan_to_num(ret, copy=False, nan=default, posinf=default, neginf=default)
    return ret


def get_logger(name: str, filepath: str = None, log_level: int = logging.INFO) -> logging.Logger:
    """Function to create a logger, or return the existing one"""
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handlers = [logging.StreamHandler()]
    if filepath:
        handlers.append(logging.FileHandler(filepath, mode="w"))
    for h in handlers:
        h.setLevel(log_level)
        h.setFormatter(formatter)
        logger.addHandler(h)
    return logger


def nullable_int(val: str) -> int:
    """Utility int checker function to accept None as
    cli argument instead of an int"""
    ret = int(val)
    if ret < 1:
        return None
    return ret


def all_subclasses(cls, recursive=False):
    """Function to return all subclasses of a given class, optionally
    searching for recursive ones."""
    if recursive:
        recursive_import(sys.modules[cls.__module__])
    return set(cls.__subclasses__()).union(
        [s for c in cls.__subclasses__() for s in all_subclasses(c)])


@dataclass
class UpdatableDataclass:
    """Base dataclass to support the update method, which updates the fields
    of the current one by looking at all the fields of the provided one"""

    def update(self, other):
        for k in fields(other):
            k = k.name
            if not hasattr(self, k):
                continue
            if isinstance(getattr(other, k), dict):
                for kk, vv in getattr(other, k).items():
                    getattr(self, k).setdefault(kk, 0)
                    getattr(self, k)[kk] += vv
            elif isinstance(getattr(other, k), list):
                setattr(self, k, getattr(self, k) + getattr(other, k))
            elif isinstance(getattr(other, k), np.ndarray):
                setattr(self, k, np.concatenate(
                    (getattr(self, k), getattr(other, k)), axis=0, dtype=getattr(self, k).dtype))
            elif is_dataclass(getattr(other, k)):
                getattr(self, k).update(getattr(other, k))
            else:
                setattr(self, k, getattr(self, k) + getattr(other, k))


def recursive_import(base_module):
    """Function to recursively import all modules within a base module"""
    if not hasattr(base_module, "__path__"):
        return

    for _, modname, ispkg in pkgutil.walk_packages(
            path=base_module.__path__,
            prefix=base_module.__name__ + ".",
            onerror=lambda x: None):

        if modname in sys.modules:
            continue

        if ispkg:
            recursive_import(modname)
        else:
            try:
                sys.modules[modname] = importlib.import_module(modname)
            except Exception as e:
                print("Exception while importing", modname, e)


def get_param_for_method(method: Callable, exclude_super_cls=None, ignore_default=False) -> Dict[str, Tuple[Any, Any]]:
    """Function to return all params and info for a specific method. Optionally, all the parameters shared with the
    provided super class are ignored."""
    if not exclude_super_cls:
        super_ones = []
    elif inspect.isclass(method):
        super_ones = inspect.signature(exclude_super_cls).parameters
    else:
        super_ones = inspect.signature(
            getattr(exclude_super_cls, method.__name__)).parameters
    return {v.name: (v.annotation if v.annotation != inspect._empty else Any, v.default)
            for v in inspect.signature(method).parameters.values() if v.name != "kwargs" and (
                v.name not in super_ones or ignore_default or (
                    not ignore_default and v.default != super_ones[v.name].default))}


def get_max_comb(tmp: Dict[str, Tuple[Any]]):
    """Return the maximum combination of key-values in a dictionary"""
    if not tmp:
        return {}
    if isinstance(tmp, (list, tuple)):
        max = tmp[0]
        for x in tmp:
            if all(x[k] >= max[k] for k in x):
                max = x
        return max
    return {k: max(v) for k, v in tmp.items()}


def get_all_dict_comb(tmp: Dict[str, Tuple[Any]]):
    """Return all combination of key-values in a dictionary"""
    import itertools
    keys, values = [], []
    for k, v in tmp.items():
        keys.append(k)
        values.append(v if isinstance(v, (tuple, list)) else [v])
    return [dict(zip(keys, v)) for v in itertools.product(*values)]


def add_param_to_parser(parser, pname, ptype, pdef, help=""):
    """Function to add a cli parameter"""
    other = {"help": help}
    if hasattr(ptype, "__origin__") and issubclass(ptype.__origin__, (Tuple, List)):
        other["nargs"] = "+"
        ptype = ptype.__args__[0]

    if ptype == int and pdef is None:
        ptype = nullable_int
    if ptype == bool:
        parser.add_argument(f'--{pname}', **other, action="store_{}".format(
            str(not pdef).lower() if pdef != inspect._empty and pdef is not None else "true"))
    elif pdef != inspect._empty:
        parser.add_argument(f'--{pname}', **other,  type=ptype, default=pdef)
    else:
        parser.add_argument(pname, **other, type=ptype)
