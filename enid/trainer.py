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
"""_
Main file for Training the desired Detection Models.
Given the combination of hyperparameters and iterable parameters
for generating models with less information (e.g., maximum number of packets P and
features extracted F), this file will invoke the train method of the
Detection Models untill all models are created. Every generated model is tested
offline against the test portion of the chosen dataset, registering results at different
granularities (i.e., for the entire dataset, for the single category of captures,
for the single pcap).
"""
import argparse
import os
from dataclasses import dataclass, field
from typing import Dict, Type

import numpy as np
import tensorflow as tf

from .datasetter import DatasetConfig, DatasetTrainConfig
from .lib.definitions import DeParams, DetectionEngine
from .lib.metrics import TrainMetric
from .lib.utility import (SplitType, add_param_to_parser, create_dir,
                          dump_json_data, get_logger, get_param_for_method,
                          load_json_data, splittable_int)

_logger = get_logger(__name__)


@dataclass
class ResultCategory(TrainMetric):
    captures: Dict[str, TrainMetric] = field(default_factory=dict)

    def __post_init__(self):
        super().__post_init__()
        for k in self.captures:
            if not isinstance(self.captures[k], TrainMetric):
                self.captures[k] = TrainMetric(**self.captures[k])


@dataclass
class ResultDataset(TrainMetric):
    categories: Dict[str, ResultCategory] = field(default_factory=dict)

    def __post_init__(self):
        super().__post_init__()
        for k in self.categories:
            if not isinstance(self.categories[k], ResultCategory):
                self.categories[k] = ResultCategory(**self.categories[k])


@dataclass
class ResultTrain(TrainMetric):
    datasets: Dict[str, ResultDataset] = field(default_factory=dict)

    def __post_init__(self):
        super().__post_init__()
        for k in self.datasets:
            if not isinstance(self.datasets[k], ResultDataset):
                self.datasets[k] = ResultDataset(**self.datasets[k])

    def update(self, test_dict: DatasetTrainConfig, pt):
        # registering results at each granularity
        if len(pt) != len(self._ypred):
            return
        i = 0
        for dataset_name, v in test_dict.datasets.items():
            self.datasets[dataset_name] = ResultDataset()
            for cat, val in v.categories.items():
                self.datasets[dataset_name].categories[cat] = ResultCategory()
                for pcap, target in val.captures.items():
                    next_i = i + target.benign.test_taken + target.malicious.test_taken
                    indexes = np.where(
                        np.logical_and(pt >= i, pt < next_i))
                    y_pred_slice = self._ypred[indexes]
                    y_test_slice = self._ytrue[indexes]
                    self.datasets[dataset_name].categories[cat].captures[pcap] = TrainMetric(
                        _threshold=[
                            (self._threshold[0][0], y_pred_slice.size)],
                        _ypred=y_pred_slice,
                        _ytrue=y_test_slice)
                    self.datasets[dataset_name].categories[cat].update(
                        self.datasets[dataset_name].categories[cat].captures[pcap])
                    i = next_i
                self.datasets[dataset_name].update(
                    self.datasets[dataset_name].categories[cat])


@dataclass
class ModelConfig:
    detection_engine: Type[DetectionEngine]
    train_params: Type[DeParams]
    split_type: SplitType = field(default=SplitType.NONE)

    def __post_init__(self):
        if isinstance(self.detection_engine, str):
            self.detection_engine = DetectionEngine.import_de(
                self.detection_engine)
        if isinstance(self.train_params, dict):
            self.train_params = self.detection_engine.de_params(
                **self.train_params)
        if isinstance(self.split_type, int):
            self.split_type = SplitType(self.split_type)

    def join(self, other: "ModelConfig") -> bool:
        if self.detection_engine != other.detection_engine or str(self.train_params) != str(other.train_params):
            return False
        return True


def train(dataset_path: str, de: Type[DetectionEngine],
          models_dir, test_dict: DatasetTrainConfig, train_params: Type[DeParams], c,
          split_type: SplitType, split_chunk):
    """Function for training a given configuration"""

    tf.keras.backend.clear_session()
    name = de.model_name(**c)
    new_path = os.path.join(models_dir, name)
    # check if configuration already trained
    if os.path.isdir(new_path) and all(
            os.path.isfile(os.path.join(new_path, x))
            for x in ("results.json", "relevance.json", "history.json", "params.json")):
        _logger.info(
            f"Model {name} already trained, skipping")
        return
    create_dir(new_path, overwrite=True)

    # calling the Detection Model specific train method
    hs, hp, best_model, xts, yts, features_holder, pt = de.train(
        dataset_path, new_path, train_params, split_type, split_chunk, **c)

    _logger.info(f"Testing {name}")
    y_pred = de.predict(best_model, xts, **hp.__dict__)

    dump_json_data(features_holder, os.path.join(new_path, "relevance.json"))
    dump_json_data(hs, os.path.join(new_path, "history.json"))
    dump_json_data(hp, os.path.join(new_path, "params.json"))

    res = ResultTrain(
        _threshold=hp.malicious_threshold,
        _ypred=y_pred, _ytrue=yts)
    res.update(test_dict, pt)

    dump_json_data(res, os.path.join(new_path, "results.json"))


def main(args_list):
    # registering cli parameters to be shown with the -h flag
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        'dataset', help='path to the dataset directory', type=str)
    parser.add_argument(
        'output', help='output name', type=str)
    parser.add_argument(
        '-s', '--split-type', help='split type', type=splittable_int, default=SplitType.NONE)

    parser.add_argument("detection_engine", help="Select the detection engine to use", type=str,
                        choices=DetectionEngine.list_all())
    parser.add_argument("rest", nargs=argparse.REMAINDER)

    args = parser.parse_args(args_list).__dict__

    de: Type[DetectionEngine] = DetectionEngine.import_de(
        args["detection_engine"])
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    add_comb_params = get_param_for_method(
        de.model_name, exclude_super_cls=DetectionEngine, ignore_default=True)
    params_for_method = {k: v for k, v in get_param_for_method(
        de.de_params).items() if k not in add_comb_params}

    # registering nested cli parameters once chosen the detection model (i.e., LucidCnn)
    # a nested -h flag is available for showing the help menu
    for pname, (ptype, pdef) in add_comb_params.items():
        parser.add_argument(
            f"--{pname}", help="Parameter for creating models", type=ptype, nargs="+",
            **{"default": tuple(range(len(de.features_holder_cls.ALLOWED), 0, -1))}
            if pname == "features" else {"required": True})

    for pname, (ptype, pdef) in params_for_method.items():
        add_param_to_parser(parser, pname, ptype, pdef,
                            "Parameter for Training")

    args.update(parser.parse_args(args["rest"]).__dict__)

    dataset_config = DatasetConfig(
        **load_json_data(os.path.join(args["dataset"], "conf.json")))

    # check if Detection Models are compatible
    if not DetectionEngine.intersect([args["detection_engine"]] +
                                     [v.detection_engine.__name__ for v in dataset_config.preprocessed_configs.values()]):
        raise ValueError("Error with the DE")

    models_config = ModelConfig(
        de, de.de_params(
            **{a: args[a] for a in params_for_method if a not in add_comb_params},
            **{a: sorted(args[a], reverse=True) for a in add_comb_params}), args["split_type"])

    models_dir = os.path.join(
        args["dataset"], args["output"], 'models')

    create_dir(models_dir, overwrite=False)

    dump_json_data(models_config, os.path.join(models_dir, "conf.json"))

    combs = models_config.train_params.all_train_combs()
    for i, c in enumerate(combs):
        train(args["dataset"], de, models_dir, dataset_config.offline,
              models_config.train_params, c, models_config.split_type, split_chunk=(i, len(combs)))
