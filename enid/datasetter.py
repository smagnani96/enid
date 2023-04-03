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
Main file for creating a Dataser from a series of preprocessed pcap files.
Pcap can belong to different datasets and categories. This program takes into
account the creation of a unified and balanced train, validation and test set
for the training and the offline testing of the generated models.
In addition, a PCAP file from the testing samples is created for the further
online testing methodology.

The algorithm for the creation of the dataset aims at taking an equally number
of benign and malicious samples. However, within the same type (e.g., malicious)
it is possible to still have a different amount of samples (e.g., 1000 ddos,
10 sqli, 1 botnet).
"""
import argparse
import math
import multiprocessing
import os
import random
import time
from copy import deepcopy
from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Type

import joblib
from pypacker.ppcap import Reader, Writer

from .lib import ATTACK_LABELS
from .lib.definitions import DetectionEngine
from .lib.identifiers import BaseKey, str_to_key
from .lib.utility import (UpdatableDataclass, all_subclasses, create_dir,
                          dump_json_data, get_logger, load_json_data)
from .preprocesser import CategoryConfig, PcapConfig, PreprocessedConfig

_logger = get_logger(__name__)


@dataclass
class SingleDatasetTestConfig(PcapConfig, UpdatableDataclass):
    categories: Dict[str, CategoryConfig] = field(default_factory=dict)

    def __post_init__(self):
        for k, v in self.categories.items():
            self.categories[k] = CategoryConfig(**v)


@dataclass
class DatasetTestConfig(PcapConfig, UpdatableDataclass):
    datasets: Dict[str, SingleDatasetTestConfig] = field(default_factory=dict)

    def __post_init__(self):
        for k, v in self.datasets.items():
            self.datasets[k] = SingleDatasetTestConfig(**v)


@dataclass
class BaseConfig(UpdatableDataclass):
    taken: int = field(default=0)
    train_taken: int = field(default=0)
    val_taken: int = field(default=0)
    test_taken: int = field(default=0)


@dataclass
class TrainBaseConfig:
    benign: BaseConfig = field(default_factory=BaseConfig)
    malicious: BaseConfig = field(default_factory=BaseConfig)


@dataclass
class TrainCategoryConfig(TrainBaseConfig):
    captures: Dict[str, TrainBaseConfig] = field(default_factory=dict)


@dataclass
class TrainDatasetConfig(TrainBaseConfig):
    categories: Dict[str, TrainCategoryConfig] = field(default_factory=dict)


@dataclass
class DatasetTrainConfig(TrainBaseConfig):
    validation_percentage: float = field(default=0.1)
    test_percentage: float = field(default=0.1)
    max_to_take: int = field(default=0)
    datasets: Dict[str, TrainDatasetConfig] = field(default_factory=dict)


@dataclass
class DatasetConfig:
    name: str = field(default="")
    time_window: int = 0
    duration: int = 0
    additional_params: Dict[str, Any] = field(default_factory=dict)
    key_cls: Type[BaseKey] = field(default=None)
    offline: DatasetTrainConfig = field(default_factory=DatasetTrainConfig)
    online: DatasetTestConfig = field(default_factory=DatasetTestConfig)
    attackers: List[Type[BaseKey]] = field(default_factory=list)
    preprocessed_configs: Dict[str, PreprocessedConfig] = field(
        default_factory=dict)
    paths: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        self.name = ""
        des = set()
        if isinstance(self.key_cls, str):
            self.key_cls = str_to_key(self.key_cls)
        for i, k in enumerate(sorted(self.preprocessed_configs.keys())):
            if isinstance(self.preprocessed_configs[k], dict):
                self.preprocessed_configs[k] = PreprocessedConfig(
                    **self.preprocessed_configs[k])
            if i != 0:
                self.name += "_"
            self.name += self.preprocessed_configs[k].family + "-" +\
                self.preprocessed_configs[k].detection_engine.__name__
            des.add(self.preprocessed_configs[k].detection_engine.__name__)
            if not self.key_cls:
                self.key_cls = self.preprocessed_configs[k].key_cls
            if not self.key_cls == self.preprocessed_configs[k].key_cls:
                raise Exception("Key cls does not match", self.key_cls,
                                self.preprocessed_configs[k].key_cls)
            if not self.time_window:
                self.time_window = self.preprocessed_configs[k].time_window
            if not self.time_window == self.preprocessed_configs[k].time_window:
                raise Exception("Time Windows does not match")
        if not DetectionEngine.intersect(des):
            raise Exception("Do not intersect")
        if isinstance(self.online, dict):
            self.online = DatasetTestConfig(**self.online)
        if isinstance(self.offline, dict):
            self.offline = DatasetTrainConfig(**self.offline)
        conf_names = list(self.preprocessed_configs.keys())
        if not all(
                self.preprocessed_configs[x].time_window == self.preprocessed_configs[conf_names[0]].time_window
                for x in conf_names):
            raise ValueError("Non son compatibili TW")
        if not all(
                self.preprocessed_configs[x].additional_params == self.preprocessed_configs[conf_names[0]].additional_params
                for x in conf_names):
            raise ValueError("Non son compatibili FL")
        for i, v in enumerate(self.attackers):
            tmp = None
            if isinstance(v, BaseKey):
                break
            if not tmp:
                tmp = next(y for y in all_subclasses(BaseKey)
                           if len(v) == len(fields(y)) and all(p.name in v for p in fields(y)))
            self.attackers[i] = tmp.create(**v)


def load_packets(pcap, dataset, category, capture):
    """Method to load all raw packets from a pcap into a buffer"""
    _logger.info(f"Started Loading packets of {pcap}")
    init_ts = 0
    all_pkts = []
    for i, (ts, pkt) in enumerate(Reader(filename=pcap)):
        if i == 0:
            init_ts = ts
        all_pkts.append((ts - init_ts, pkt, dataset, category, capture))
    _logger.info(f"Finished Loading packets of {pcap}")
    return all_pkts


def async_combined(unordered, target_dir):
    """Method to sort all packets belonging to the provided pcaps by arrival time and
    creating a unified capture file"""
    _logger.info("Start Combined Async load")
    pkts = []
    [pkts.extend(x) for x in unordered]
    pkts = sorted(pkts, key=lambda x: x[0])
    tmp = []
    with Writer(filename=os.path.join(target_dir, "combined.pcap")) as w:
        new_ts = time.time_ns()
        for i, x in enumerate(pkts):
            w.write(x[1], ts=new_ts+x[0])
            tmp.append((x[2], x[3], x[4]))
            if i % 50000 == 0:
                _logger.info(f"Report Combined Async Load {i} {len(pkts)}")
    with open(os.path.join(target_dir, "combined.joblib"), "wb") as fp:
        joblib.dump(tmp, fp)
    _logger.info("Finished Combined Async load")
    return pkts[-1][0]


def async_join(conf: DatasetTrainConfig, preprocessed: Dict[str, PreprocessedConfig],
               paths: Dict[str, str], target_dir, de: DetectionEngine):
    """Method to joining all portions of train, validation and test processed data into the final one"""
    for dataset, v in conf.datasets.items():
        for category, vv in v.categories.items():
            for capture, vvv in vv.captures.items():
                for label, ttype in enumerate(["benign", "malicious"]):
                    t: BaseConfig = getattr(vvv, ttype)
                    if not t.taken:
                        continue
                    spath = os.path.join(paths[dataset], category, capture)
                    available = getattr(
                        preprocessed[dataset].categories[category].captures[capture], ttype)
                    indexes = sorted(random.sample(range(available), t.taken))

                    de.append_to_dataset(spath, os.path.join(
                        target_dir, "train"), ttype, label, indexes[:t.train_taken])
                    de.append_to_dataset(spath, os.path.join(
                        target_dir, "validation"), ttype, label, indexes[t.train_taken:t.val_taken])
                    de.append_to_dataset(spath, os.path.join(
                        target_dir, "test"), ttype, label, indexes[t.train_taken+t.val_taken:])


def main(args_list):
    # registering cli args
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        'preprocessed', help='preprocessed directories', type=str, nargs="+")
    parser.add_argument(
        '-tp', '--test-percentage', help='percentage of test in dataset', type=float, default=0.1)
    parser.add_argument(
        '-vp', '--validation-percentage', help='percentage of validation within the entire train set',
        type=float, default=0.1)
    parser.add_argument(
        '-p', '--parallel', help='number of parallel executions', type=int, default=os.cpu_count())
    args = parser.parse_args(args_list).__dict__

    paths = {}
    tmp_confs = {}
    attackers_list = []
    detection_engine: DetectionEngine = None

    for c in set(args["preprocessed"]):
        tmp = PreprocessedConfig(
            **load_json_data(os.path.join(c, "conf.json")))
        tmp_confs[tmp.family] = tmp
        detection_engine = tmp.detection_engine
        paths[tmp.family] = c
        attackers_list += ATTACK_LABELS[tmp.family](
            tmp.captures_config.path)

    conf: DatasetConfig = DatasetConfig(
        preprocessed_configs=tmp_confs, attackers=attackers_list,
        additional_params=tmp.additional_params, paths=paths)

    target_dir = os.path.join("datasets", conf.name)
    create_dir(target_dir, overwrite=False)

    # Chosing portions for the online simulation according to their maximum number
    # of benign and malicious samples (which are maximised)
    test_pcaps = []
    cop = deepcopy(conf.preprocessed_configs)
    for dataset_name in cop.keys():
        conf.online.datasets[dataset_name] = SingleDatasetTestConfig()
        for cat, vals in conf.preprocessed_configs[dataset_name].categories.items():
            conf.online.datasets[dataset_name].categories[cat] = CategoryConfig(
            )
            chosen = max(vals.captures, key=lambda x: getattr(
                vals.captures[x], "malicious" if vals.malicious else "benign"))
            tmp = cop[dataset_name].categories[cat].captures.pop(chosen)
            conf.online.datasets[dataset_name].categories[cat].captures[chosen] = tmp
            test_pcaps.append((os.path.join(
                conf.preprocessed_configs[dataset_name].captures_config.path, cat, chosen),
                cop[dataset_name].family, cat, chosen))
            conf.online.datasets[dataset_name].categories[cat].update(tmp)
            conf.online.datasets[dataset_name].update(tmp)
            conf.online.update(tmp)

    with multiprocessing.Pool(maxtasksperchild=1, processes=args["parallel"]) as pool:
        pkts = pool.starmap_async(load_packets, test_pcaps)

        # Creating balanced train, validation and test portion for training
        tmp = [(vvv.benign, vvv.malicious) for v in cop.values() for vv in v.categories.values()
               for vvv in vv.captures.values()]
        conf.offline.max_to_take = min(
            sum([v[0] for v in tmp]), sum([v[1] for v in tmp]))
        conf.offline.test_percentage = args["test_percentage"]
        conf.offline.validation_percentage = args["validation_percentage"]
        for ttype in ("benign", "malicious"):
            asd = {(k, kk): sum([getattr(vvv, ttype) for vvv in vv.captures.values()])
                   for k, v in cop.items() for kk, vv in v.categories.items()}
            asd = {k: v for k, v in asd.items() if v}
            asd = {k: asd[k] for k in sorted(asd, key=asd.get)}
            if not asd:
                continue
            take_for_each_cat = math.floor(conf.offline.max_to_take/len(asd))
            for ii, (dataset_name, cat) in enumerate(asd.keys()):
                take_for_each_pcap = math.floor(
                    take_for_each_cat/len(cop[dataset_name].categories[cat].captures))
                if dataset_name not in conf.offline.datasets:
                    conf.offline.datasets[dataset_name] = TrainDatasetConfig()
                if cat not in conf.offline.datasets[dataset_name].categories:
                    conf.offline.datasets[dataset_name].categories[cat] = TrainCategoryConfig(
                    )
                for i, (name, vals) in enumerate({k: cop[dataset_name].categories[cat].captures[k] for k in
                                                  sorted(cop[dataset_name].categories[cat].captures,
                                                  key=lambda x: getattr(cop[dataset_name].categories[cat].captures[x], ttype))}
                                                 .items()):
                    if name not in conf.offline.datasets[dataset_name].categories[cat].captures:
                        conf.offline.datasets[dataset_name].categories[cat].captures[name] = TrainBaseConfig(
                        )
                    taken = min(getattr(vals, ttype), take_for_each_pcap)
                    test_start = math.ceil(
                        taken * (1 - conf.offline.test_percentage))
                    val_start = math.ceil(
                        test_start * (1 - conf.offline.validation_percentage))

                    tmp = BaseConfig(
                        taken, val_start, test_start - val_start, taken - test_start)
                    setattr(
                        conf.offline.datasets[dataset_name].categories[cat].captures[name], ttype, tmp)

                    getattr(
                        conf.offline.datasets[dataset_name].categories[cat], ttype).update(tmp)
                    getattr(
                        conf.offline.datasets[dataset_name], ttype).update(tmp)
                    getattr(conf.offline, ttype).update(tmp)

                    if i+2 == len(cop[dataset_name].categories[cat].captures):
                        take_for_each_pcap = take_for_each_cat - \
                            getattr(
                                conf.offline.datasets[dataset_name].categories[cat], ttype).taken
                    elif i+1 != len(cop[dataset_name].categories[cat].captures) and taken < take_for_each_pcap:
                        take_for_each_pcap = math.floor(
                            (take_for_each_cat-getattr(conf.offline.datasets[dataset_name].categories[cat], ttype).taken) /
                            (len(cop[dataset_name].categories[cat].captures)-i-1))
                        _logger.info("Taken less for pcap {} {}, increasing to {}".format(
                            taken, name, take_for_each_pcap))

                taken_so_far = sum(
                    getattr(t, ttype).taken for t in conf.offline.datasets.values())
                if ii+2 == len(asd):
                    take_for_each_cat = conf.offline.max_to_take-taken_so_far
                elif ii+1 != len(asd) and getattr(conf.offline.datasets[dataset_name].categories[cat],
                                                  ttype).taken < take_for_each_cat:
                    take_for_each_cat = math.floor((conf.offline.max_to_take-taken_so_far) /
                                                   (len(asd)-sum(len(p.categories) for p in conf.offline.datasets.values())))
                    _logger.info("Taken less {} for cat {}, increasing to {}".format(
                        getattr(conf.offline.datasets[dataset_name].categories[cat], ttype).taken, cat, take_for_each_cat))

        tt = pool.apply_async(async_join, (conf.offline, conf.preprocessed_configs, paths,
                                           target_dir, detection_engine))
        pkts = pkts.get()
        d = pool.apply_async(async_combined, (pkts, target_dir))

        tt.get()
        conf.duration = d.get()

    _logger.info("Dumping configuration with updated stats")
    dump_json_data(conf, os.path.join(target_dir, "conf.json"))
