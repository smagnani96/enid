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
"""
Main file for registering new Detection Engine to a previous ENID installation
"""
import argparse
import os
import shutil

from .lib.definitions import DetectionEngine
from .lib.utility import get_logger, snake_to_camel

_logger = get_logger(__name__)


def main(args_list):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('path',
                        help='path to the de to register', type=str)
    args = parser.parse_args(args_list).__dict__

    args["path"] = os.path.normpath(args["path"])

    if os.path.isfile(args["path"]) and args["path"].endswith(".py"):
        de_name = os.path.basename(args["path"]).replace(".py", "")
    elif os.path.isdir(args["path"]) and os.path.isfile(os.path.join(args["path"], "__init__.py")):
        de_name = os.path.basename(args["path"])
    else:
        raise RuntimeError("Unsupported format for registering plugin")
    de_name_camelised = snake_to_camel(de_name)

    dest_path = os.path.join(os.path.dirname(
        __file__), "lib", "engines", de_name)

    _logger.info(
        f"Copying Detection Engine {de_name_camelised} to directory {dest_path}")
    shutil.copytree(args["path"], dest_path, dirs_exist_ok=True)
    try:
        _logger.info("Checking validity...")
        DetectionEngine.import_de(de_name_camelised)
        _logger.info("Engine successfully installed!")
    except Exception as e:
        _logger.info(f"Invalid Engine, removing. Why: {e}")
        shutil.rmtree(dest_path)
