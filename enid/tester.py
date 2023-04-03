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
Main file for Testing the generated models' configurations.
This file executes in parallel on different processes the online
testing methodology with the Traffic Filter, the Feature Extractor,
and the Classifier all active. For each configuration, the entire pipeline
is adjusted accordingly to the set of parameters involved.
"""
import argparse
import multiprocessing
import os

from dataclasses import dataclass, field

from .lib.definitions import TestType, TrafficAnalyser, DebugLevel
from .lib.utility import create_dir, dump_json_data, load_json_data
from .trainer import DatasetConfig, ModelConfig


@dataclass
class TestConfig:
    debug: DebugLevel = DebugLevel.NONE
    enforce_timewindows_delay: int = field(default=1)
    sessions: int = None
    is_throughput: bool = False
    is_adaptiveness_enabled: bool = field(default=False)

    def __post_init__(self):
        if isinstance(self.debug, int):
            self.debug = DebugLevel(self.debug)


def _run_async_adaptiveness(models_conf: ModelConfig, dataset_conf: DatasetConfig, test_conf: TestConfig, basedir, c):

    name = models_conf.detection_engine.model_name(**c)
    create_dir(os.path.join(basedir, name))
    t = models_conf.detection_engine.traffic_analyser_cls(
        models_conf.detection_engine,
        dataset_conf.attackers,
        dataset_conf.time_window,
        dataset_conf.key_cls,
        TestType.THROUGHPUT if test_conf.is_throughput else TestType.NORMAL,
        models_dir=os.path.join(basedir, os.pardir, "models"),
        debug=test_conf.debug,
        enforce_timewindows_delay=test_conf.enforce_timewindows_delay,
        is_adaptiveness_enabled=test_conf.is_adaptiveness_enabled,
        sessions_per_timewindow=test_conf.sessions,
        **c)
    TrafficAnalyser.generate_packets(os.path.join(
        basedir, os.pardir, os.pardir, "combined.pcap"), t)
    dump_json_data(t.results, os.path.join(basedir, name, "results.json"))
    if test_conf.debug != DebugLevel.NONE:
        dump_json_data(t.debug_data, os.path.join(
            basedir, name, "history.json"))


def main(args_list):
    # registering cli parameters, which can be shown with the -h
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        'de_models_dir', help='path to the model directory containing the models to test')
    parser.add_argument(
        '-d', '--debug', help='debug mode', type=int, default=0)
    parser.add_argument(
        '-dd', '--detection-delay', help='number of time windows to wait before performing detection+mitigation',
        type=int, default=0)
    parser.add_argument(
        '-s', '--sessions', help='number of monitored sessions at once', type=int, default=None)
    parser.add_argument(
        '-t', '--throughput', help='test throughput', action="store_true")
    parser.add_argument(
        '-a', '--adaptiveness', help='enable adaptiveness to auto adjust', action="store_true")
    parser.add_argument(
        '-p', '--parallel', help='number of parallel executions', type=int, default=os.cpu_count())

    args = parser.parse_args(args_list).__dict__

    # loading configurations
    models_config = ModelConfig(
        **load_json_data(os.path.join(args["de_models_dir"], "models", "conf.json")))

    dataset_config = DatasetConfig(
        **load_json_data(os.path.join(args["de_models_dir"], os.pardir, "conf.json")))

    test_config = TestConfig(
        args["debug"], args["detection_delay"], args["sessions"], args["throughput"], args["adaptiveness"])

    dir_basename = os.path.join(
        args["de_models_dir"], "throughput_test" if test_config.is_throughput else "normal_test")

    create_dir(dir_basename, overwrite=False)

    dump_json_data(test_config, os.path.join(dir_basename, "conf.json"))

    # create an async process to test each configuration standalone
    with multiprocessing.Pool(maxtasksperchild=1, processes=args["parallel"]) as pool:
        pool.starmap(_run_async_adaptiveness, [
            (models_config, dataset_config, test_config, dir_basename, c)
            for c in models_config.train_params.all_train_combs()])
