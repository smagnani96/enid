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
Main file for Processing input network captures into data according to the specified
Detection Model and parameters. This creates the files ready to be used for the creation of the dataset.
For each capture/category/dataset, this program records the n° of benign and malicious samples and
the respective network packets, information used later during the dataset creation.
A key for grouping sessions needs to be chosen, whether considering only L3 or also L4.
"""
import argparse
import multiprocessing
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Type

from .lib import ATTACK_LABELS
from .lib.definitions import DetectionEngine, TestType, TrafficAnalyser
from .lib.identifiers import BaseKey, str_to_key
from .lib.utility import (UpdatableDataclass, add_param_to_parser,
                          all_subclasses, create_dir, dump_json_data,
                          get_logger, get_param_for_method, load_json_data)
from .splitter import CaptureConfig

_logger = get_logger(__name__)


@dataclass
class PcapConfig:
    benign: int = 0
    malicious: int = 0
    unique_benign: int = 0
    unique_malicious: int = 0
    benign_packets: int = 0
    malicious_packets: int = 0


@dataclass
class CategoryConfig(PcapConfig, UpdatableDataclass):
    captures: Dict[str, PcapConfig] = field(default_factory=dict)

    def __post_init__(self):
        for k in list(self.captures.keys()):
            self.captures[k] = PcapConfig(**self.captures[k])


@dataclass
class PreprocessedConfig:
    family: str
    time_window: float
    key_cls: Type[BaseKey]
    detection_engine: Type[DetectionEngine]
    additional_params: Dict[str, Any]
    categories: Dict[str, CategoryConfig]
    captures_config: CaptureConfig

    def __post_init__(self):
        if isinstance(self.captures_config, dict):
            self.captures_config = CaptureConfig(**self.captures_config)
        for k in list(self.categories.keys()):
            self.categories[k] = CategoryConfig(**self.categories[k])
        if isinstance(self.detection_engine, str):
            self.detection_engine = DetectionEngine.import_de(
                self.detection_engine)
        if isinstance(self.key_cls, str):
            self.key_cls = str_to_key(self.key_cls)


def process_pcap(target_dir, cap, dataset, cat, pcap, time_window, additional_params,
                 attackers, de: Type[DetectionEngine], key_cls: Type[BaseKey]):
    """Function for parsing a pcap"""
    target_ds = os.path.join(target_dir, cat, pcap)
    pcap_file = os.path.join(cap, cat, pcap)
    if not os.path.isfile(pcap_file):
        return None

    t = de.traffic_analyser_cls(
        de, attackers, time_window, key_cls, TestType.PROCESSING,
        dump_path=target_ds,
        **additional_params)

    _logger.info(f"Starting processing {pcap_file}")
    # Creating a fake traffic analysers with no Filtering and Classificator components.
    TrafficAnalyser.generate_packets(pcap_file, t, pcap_file, labels=(dataset, cat, pcap))
    _logger.info(f"Finished processing {pcap_file}")
    p = PcapConfig(t.processing_stats.tot_benign, t.processing_stats.tot_malicious,
                   t.processing_stats.unique_benign, t.processing_stats.unique_malicious,
                   t.processing_stats.tot_benign_packets, t.processing_stats.tot_malicious_packets)
    return target_dir, cat, pcap, p


def main(args_list):
    # registering cli args, shown with the -h flag
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('time_window',
                        help='time window of interest in nanoseconds', type=int)
    parser.add_argument('key',
                        help='Session Key to use', choices=[x.__name__ for x in all_subclasses(BaseKey)], type=str)
    parser.add_argument(
        '-p', '--parallel', help='number of parallel executions', type=int, default=os.cpu_count())
    parser.add_argument("detection_engine", help="Select the detection engine to use", type=str,
                        choices=DetectionEngine.list_all())
    parser.add_argument("rest", nargs=argparse.REMAINDER)

    args = parser.parse_args(args_list).__dict__

    de: Type[DetectionEngine] = DetectionEngine.import_de(
        args["detection_engine"])

    # registering Detection Model - specific arguments
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    params_for_method = get_param_for_method(
        de.model_name, exclude_super_cls=DetectionEngine)
    for pname, (ptype, pdef) in params_for_method.items():
        add_param_to_parser(parser, pname, ptype, pdef, "Parameter")
    parser.add_argument(
        'captures', help='capture directories', type=str, nargs="+")
    args.update(parser.parse_args(args["rest"]).__dict__)

    additional_params = {k: args[k] for k in params_for_method}

    key_cls = str_to_key(args["key"])
    star_tasks = []
    prep_configs: Dict[str, PreprocessedConfig] = {}

    for cap in args["captures"]:
        dataset = os.path.basename(os.path.normpath(cap))
        output_name = "{}-{}t-{}".format(
            de.model_name(**additional_params), args["time_window"], dataset)
        target_dir = os.path.join(
            "preprocessed", args["detection_engine"], output_name)
        create_dir(target_dir, overwrite=False)

        malicious = ATTACK_LABELS[dataset](cap)

        capture_conf = load_json_data(
            os.path.join(cap, "conf.json"), fail=False)
        if not capture_conf:
            capture_conf = CaptureConfig(path=cap)

        prep_configs[target_dir] = PreprocessedConfig(
            key_cls=key_cls,
            family=dataset,
            additional_params=additional_params, time_window=args["time_window"],
            detection_engine=args["detection_engine"], categories={},
            captures_config=capture_conf)

        for cat in os.listdir(cap):
            if not os.path.isdir(os.path.join(cap, cat)):
                continue
            create_dir(os.path.join(target_dir, cat), overwrite=False)
            for pcap in os.listdir(os.path.join(cap, cat)):
                if not pcap.endswith(".pcap"):
                    continue
                star_tasks.append((target_dir, cap, dataset, cat, pcap, args["time_window"],
                                   additional_params, malicious, de, key_cls))

    with multiprocessing.Pool(maxtasksperchild=1, processes=args["parallel"]) as pool:
        # concatenate all results of all launched processes
        for ret in pool.starmap(process_pcap, star_tasks):
            if not ret:
                continue
            td, cat, pcap, p = ret
            if cat not in prep_configs[td].categories:
                prep_configs[td].categories[cat] = CategoryConfig()
            prep_configs[td].categories[cat].captures[pcap] = p
            prep_configs[td].categories[cat].update(p)

    for td, val in prep_configs.items():
        _logger.info(f"Dumping {td} configuration with updated pcaps stats")
        dump_json_data(val, os.path.join(td, "conf.json"))
