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
import pickle
import random
import time
from copy import deepcopy
from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Type

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
    duration: int = 0
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

    def __post_init__(self):
        if isinstance(self.benign, dict):
            self.benign = BaseConfig(**self.benign)
        if isinstance(self.malicious, dict):
            self.malicious = BaseConfig(**self.malicious)


@dataclass
class TrainCategoryConfig(TrainBaseConfig):
    captures: Dict[str, TrainBaseConfig] = field(default_factory=dict)

    def __post_init__(self):
        for k, v in self.captures.items():
            if isinstance(v, dict):
                self.captures[k] = TrainBaseConfig(**v)


@dataclass
class TrainDatasetConfig(TrainBaseConfig):
    categories: Dict[str, TrainCategoryConfig] = field(default_factory=dict)

    def __post_init__(self):
        for k, v in self.categories.items():
            if isinstance(v, dict):
                self.categories[k] = TrainCategoryConfig(**v)


@dataclass
class DatasetTrainConfig(TrainBaseConfig):
    validation_percentage: float = field(default=0.1)
    test_percentage: float = field(default=0.1)
    max_to_take: int = field(default=0)
    datasets: Dict[str, TrainDatasetConfig] = field(default_factory=dict)

    def __post_init__(self):
        for k, v in self.datasets.items():
            if isinstance(v, dict):
                self.datasets[k] = TrainDatasetConfig(**v)


@dataclass
class DatasetConfig:
    name: str = field(default="")
    time_window: int = 0
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
            self.name += self.preprocessed_configs[k].family + "-"
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
            if i + 1 == len(self.preprocessed_configs):
                self.name += self.preprocessed_configs[k].detection_engine.__name__
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
                _logger.info(f"Report Combined Async Load {100*i/len(pkts)}%")
    with open(os.path.join(target_dir, "combined.pickle"), "wb") as fp:
        pickle.dump(tmp, fp)
    _logger.info("Finished Combined Async load")


def async_join(conf: DatasetTrainConfig, preprocessed: Dict[str, PreprocessedConfig],
               paths: Dict[str, str], target_dir, de: DetectionEngine):
    """Method to joining all portions of train, validation and test processed data into the final one"""
    _logger.info("Async Join Start")
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
                    indexes = random.sample(range(available), t.taken)
                    de.append_to_dataset(spath, target_dir, ttype, label,
                                         indexes[:t.train_taken],
                                         indexes[t.train_taken:t.train_taken +
                                                 t.val_taken],
                                         indexes[t.train_taken+t.val_taken:])
    _logger.info("Async Join End")


def main(args_list):
    # registering cli args
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-b', '--benign', help='preprocessed directories with benign flows', type=str, nargs="+", required=True)
    parser.add_argument(
        '-m', '--malicious', help='preprocessed directories with malicious flows', type=str, nargs="+", required=True)
    parser.add_argument(
        '-pc', '--per-category', help='per category creation of the dataset', action="store_true")
    parser.add_argument(
        '-no', '--no-online', help='no online test', action="store_true")
    parser.add_argument(
        '-tp', '--test-percentage', help='percentage of test in dataset', type=float, default=0.1)
    parser.add_argument(
        '-vp', '--validation-percentage', help='percentage of validation within the entire train set',
        type=float, default=0.1)
    parser.add_argument(
        '-p', '--parallel', help='number of parallel executions', type=int, default=os.cpu_count())
    args = parser.parse_args(args_list).__dict__

    conf: DatasetConfig = DatasetConfig()
    conf.offline.test_percentage = args["test_percentage"]
    conf.offline.validation_percentage = args["validation_percentage"]
    detection_engine: DetectionEngine = None

    for c in set(args["benign"] + args["malicious"]):
        tmp = PreprocessedConfig(
            **load_json_data(os.path.join(c, "conf.json")))
        conf.preprocessed_configs[tmp.family] = tmp
        detection_engine = tmp.detection_engine
        conf.paths[tmp.family] = c
        conf.attackers += ATTACK_LABELS[tmp.family](
            tmp.captures_config.path)
        conf.additional_params = tmp.additional_params
    conf.__post_init__()

    target_dir = os.path.join(
        "datasets", conf.name, "{}-{}".format("percategory" if args["per_category"] else "combined",
                                              "offline" if args["no_online"] else "online"))
    create_dir(target_dir, overwrite=False)

    # Chosing portions for the online simulation according to their maximum number
    # of benign and malicious samples (which are maximised)
    test_pcaps = []
    cop = deepcopy(conf.preprocessed_configs)
    if not args["no_online"]:
        for ttype in ("benign", "malicious"):
            for dpath in args[ttype]:
                dataset_name = next(
                    k for k, v in conf.paths.items() if v == dpath)
                conf.online.datasets[dataset_name] = SingleDatasetTestConfig()
                for cat, vals in conf.preprocessed_configs[dataset_name].categories.items():
                    conf.online.datasets[dataset_name].categories[cat] = CategoryConfig(
                    )
                    chosen = max(vals.captures, key=lambda x: getattr(
                        vals.captures[x], ttype))
                    tmp = cop[dataset_name].categories[cat].captures.pop(
                        chosen)
                    conf.online.datasets[dataset_name].categories[cat].captures[chosen] = tmp
                    test_pcaps.append((os.path.join(
                        conf.preprocessed_configs[dataset_name].captures_config.path, cat, chosen),
                        cop[dataset_name].family, cat, chosen))
                    conf.online.datasets[dataset_name].categories[cat].update(
                        tmp)
                    conf.online.datasets[dataset_name].update(tmp)
                    conf.online.update(tmp)

    with multiprocessing.Pool(maxtasksperchild=1, processes=args["parallel"]) as pool:
        pkts = None
        tasks = []
        if not args["no_online"]:
            pkts = pool.starmap_async(load_packets, test_pcaps)

        if not args["per_category"]:
            # Creating balanced train, validation and test portion for training
            tmp = [(vvv.benign, vvv.malicious) for v in cop.values() for vv in v.categories.values()
                   for vvv in vv.captures.values()]
            conf.offline.max_to_take = min(
                sum([v[0] for v in tmp]), sum([v[1] for v in tmp]))
            for ttype in ("benign", "malicious"):
                asd = {}
                for dpath in args[ttype]:
                    dname = next(
                        k for k, v in conf.paths.items() if v == dpath)
                    asd.update({(dname, kk): sum([getattr(vvv, ttype) for vvv in vv.captures.values()])
                                for kk, vv in cop[dname].categories.items()})
                asd = {k: v for k, v in asd.items() if v}
                asd = {k: asd[k] for k in sorted(asd, key=asd.get)}
                macina(conf, asd, cop, ttype)

            tasks.append(pool.apply_async(async_join, (conf.offline, conf.preprocessed_configs, conf.paths,
                                                       target_dir, detection_engine)))
            if not args["no_online"]:
                conf.online.duration = max(vvv.duration for v in conf.online.datasets.values(
                ) for vv in v.categories.values() for vvv in vv.captures.values())
            _logger.info("Dumping configuration with updated stats")
            dump_json_data(conf, os.path.join(target_dir, "conf.json"))
            if not args["no_online"]:
                pkts = pkts.get()
                tasks.append(pool.apply_async(
                    async_combined, (pkts, target_dir)))
        else:
            asd = {}
            for dpath in args["benign"]:
                dname = next(k for k, v in conf.paths.items() if v == dpath)
                asd.update({(dname, kk): sum([vvv.benign for vvv in vv.captures.values()])
                            for kk, vv in cop[dname].categories.items()})
            for dpath in args["malicious"]:
                dname = dataset_name = next(
                    k for k, v in conf.paths.items() if v == dpath)
                for cat, v in conf.preprocessed_configs[dname].categories.items():
                    confi: DatasetConfig = deepcopy(conf)
                    confi.offline.max_to_take = min(
                        v.malicious, sum(v for v in asd.values()))
                    macina(confi, asd, cop, "benign")
                    macina(confi, {(dname, cat): v.malicious},
                           cop, "malicious")
                    _logger.info("Dumping configuration with updated stats")
                    ts = os.path.join(target_dir, dname, cat)
                    create_dir(ts)
                    dump_json_data(confi, os.path.join(ts, "conf.json"))
                    tasks.append(pool.apply_async(async_join, (confi.offline, conf.preprocessed_configs, conf.paths,
                                                               ts, detection_engine)))
        _logger.info("Waiting for last tasks ...")
        for t in tasks:
            t.get()


def macina(conf: DatasetConfig, asd, cop, ttype):
    take_for_each_cat = math.floor(conf.offline.max_to_take/len(asd))
    so_so_far = 0
    for ii, (dataset_name, cat) in enumerate(asd.keys()):
        take_for_each_pcap = math.floor(
            take_for_each_cat/len(cop[dataset_name].categories[cat].captures))
        if dataset_name not in conf.offline.datasets:
            conf.offline.datasets[dataset_name] = TrainDatasetConfig()
        if cat not in conf.offline.datasets[dataset_name].categories:
            conf.offline.datasets[dataset_name].categories[cat] = TrainCategoryConfig(
            )

        so_far = 0
        for i, (name, vals) in enumerate(sorted(cop[dataset_name].categories[cat].captures.items(),
                                                key=lambda x: getattr(x[1], ttype))):
            if name not in conf.offline.datasets[dataset_name].categories[cat].captures:
                conf.offline.datasets[dataset_name].categories[cat].captures[name] = TrainBaseConfig(
                )
            taken = min(getattr(vals, ttype), take_for_each_pcap)
            so_far += taken
            if taken != take_for_each_pcap and i+1 != len(conf.preprocessed_configs[dataset_name].categories[cat].captures):
                take_for_each_pcap = math.floor((take_for_each_cat - so_far) / (len(
                    conf.preprocessed_configs[dataset_name].categories[cat].captures) - i - 1))
            test_start = math.floor(
                taken * (1 - conf.offline.test_percentage))
            val_start = math.floor(
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
        so_so_far += so_far
        if so_far != take_for_each_cat and ii+1 != len(asd):
            take_for_each_cat = math.floor(
                (conf.offline.max_to_take - so_so_far) / (len(asd)-ii-1))
