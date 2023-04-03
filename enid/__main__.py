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
import argparse
import importlib
import os

from .lib.definitions import DetectionEngine
from .lib.utility import camel_to_snake, silence, set_seed

if __name__ == '__main__':
    silence()
    set_seed()
    de_with_main_list = DetectionEngine.list_all(only_main=True)
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("operation", help="Select the operation to perform", type=str,
                        choices=de_with_main_list + [x.replace(".py", "") for x in os.listdir(os.path.dirname(__file__))
                                                     if not x.startswith("_") and x.endswith(".py")
                                                     and os.path.isfile(os.path.join(os.path.dirname(__file__), x))])
    parser.add_argument("rest", nargs=argparse.REMAINDER)
    args = parser.parse_args().__dict__
    if args["operation"] not in de_with_main_list:
        mod = importlib.import_module(".{}".format(
            args["operation"]), package=__package__)
        mod.main(args["rest"])
    else:
        args["operation"] = camel_to_snake(args["operation"])
        parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument("action", help="Main action", type=str,
                            choices=[x.replace(".py", "") for x in os.listdir(
                                os.path.join(os.path.dirname(__file__), "lib", "engines", args["operation"], "main"))
                                if not x.startswith("_") and x.endswith(".py")
                                and os.path.isfile(
                                os.path.join(os.path.dirname(__file__), "lib", "engines", args["operation"], "main", x))])
        parser.add_argument("rest", nargs=argparse.REMAINDER)
        args_nested = parser.parse_args(args["rest"]).__dict__
        mod = importlib.import_module(".lib.engines.{}.main.{}".format(
            args["operation"], args_nested["action"]), package=__package__)
        mod.main(args_nested["rest"])
