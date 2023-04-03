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
"""File for defining metrics used within the offline and online test"""
from dataclasses import dataclass, field, fields
from enum import Enum
from typing import Tuple, Union, List, Dict, Type

import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, confusion_matrix,
                             f1_score, log_loss, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)

from .utility import CDataJSONEncoder, UpdatableDataclass, safe_division


class ComputationalRequirements(Enum):
    """Enumeration for representing the computational requirements of features and key fields"""
    REQUIRED_L2 = 3
    REQUIRED_L3 = REQUIRED_L2 + 3
    REQUIRED_L4 = REQUIRED_L3 + 5
    HASH_COMPUTATION = 3
    BASE_MATH_OP = 1
    ENHANCED_MATH_OP = 2
    TIMER = 5

    @staticmethod
    def requirements_to_cost(requirements_lists: Tuple["ComputationalRequirements"], ignore_depth: bool = False):
        cost = 0

        if ignore_depth is not True:
            if isinstance(ignore_depth, bool):
                target = requirements_lists
            else:
                from .definitions import BaseKey
                ignore_depth: BaseKey = ignore_depth
                target = tuple(x for x in (ComputationalRequirements.REQUIRED_L4,
                                           ComputationalRequirements.REQUIRED_L3,
                                           ComputationalRequirements.REQUIRED_L2)
                               if x not in ignore_depth.computational_requirements)

            if ComputationalRequirements.REQUIRED_L4 in target:
                cost += ComputationalRequirements.REQUIRED_L4.value
            elif ComputationalRequirements.REQUIRED_L3 in target:
                cost += ComputationalRequirements.REQUIRED_L3.value
            elif ComputationalRequirements.REQUIRED_L2 in target:
                cost += ComputationalRequirements.REQUIRED_L2.value

        for y in (ComputationalRequirements.HASH_COMPUTATION, ComputationalRequirements.BASE_MATH_OP,
                  ComputationalRequirements.ENHANCED_MATH_OP, ComputationalRequirements.TIMER):
            cost += y.value*requirements_lists.count(y)

        return cost


@dataclass
class TrainMetric:
    """Class for holding the train metrics"""
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0

    tpr: float = 0
    fpr: float = 0
    tnr: float = 0
    fnr: float = 0

    log_loss: float = 0
    gmean: float = 0
    f1_score: float = 0
    accuracy: float = 0
    balanced_accuracy: float = 0
    roc_auc: float = 0
    recall: float = 0
    precision: float = 0
    average_precision: float = 0

    best_roc_threshold: float = 0
    best_roc_gmean: float = 0
    best_precision_recall_threshold: float = 0
    best_precision_recall_f1score: float = 0

    _threshold: List[Tuple[float, int]] = field(
        default_factory=list, repr=False)
    _ypred: np.ndarray = field(default=np.array(
        [], dtype=np.float64), repr=False)
    _ytrue: np.ndarray = field(default=np.array(
        [], dtype=np.float64), repr=False)

    def to_json(self):
        """Method to dump the class in a json-style"""
        e = CDataJSONEncoder()
        return {k.name: e.default(getattr(self, k.name)) for k in fields(self) if k.repr}

    def __post_init__(self):
        """Method called after initialisation. If the class is not empty,
        this method computes all the metrics."""
        if not isinstance(self._threshold, list):
            self._threshold = [(self._threshold, self._ytrue.size)]

        if self._ypred.size == 0 or self._ytrue.size == 0:
            return

        fpr, tpr, thresholds = roc_curve(self._ytrue, self._ypred)
        # calculate the g-mean for each threshold
        gmeans = np.sqrt(tpr * (1-fpr))
        np.nan_to_num(gmeans, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        # locate the index of the largest g-mean
        ix = np.argmax(gmeans)
        self.best_roc_threshold, self.best_roc_gmean = thresholds[ix], gmeans[ix]
        precision, recall, thresholds = precision_recall_curve(
            self._ytrue, self._ypred)
        # convert to f score
        fscore = (2 * precision * recall) / (precision + recall)
        np.nan_to_num(fscore, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        # locate the index of the largest f score
        ix = np.argmax(fscore)
        self.best_precision_recall_threshold, self.best_precision_recall_f1score = thresholds[
            ix], fscore[ix]

        self.log_loss = log_loss(self._ytrue, self._ypred, labels=[0, 1])
        ypred = np.copy(self._ypred)

        # Since it might be possible that the threshold changed during the time,
        # here we make sure to use the right threshold for the various
        # bunch of data
        prev_index = 0
        for (t, index) in self._threshold:
            if prev_index - index != 0:
                ypred[prev_index:index] = ypred[prev_index:index] > t
                prev_index = index

        self.tn, self.fp, self.fn, self.tp = confusion_matrix(
            self._ytrue, ypred, labels=[0, 1]).ravel()
        self.tpr = safe_division(self.tp, self.tp + self.fn, default=0.0)
        self.tnr = safe_division(self.tn, self.tn + self.fp, default=0.0)
        self.fpr = safe_division(self.fp, self.fp + self.tn, default=0.0)
        self.fnr = safe_division(self.fn, self.fn + self.tp, default=0.0)

        self.gmean = np.sqrt(self.tpr * (1-self.fpr))
        self.f1_score = f1_score(self._ytrue, ypred, labels=[0, 1])
        self.accuracy = accuracy_score(self._ytrue, ypred)
        self.balanced_accuracy = balanced_accuracy_score(self._ytrue, ypred)
        try:
            self.roc_auc = roc_auc_score(self._ytrue, ypred, labels=[0, 1])
        except ValueError:
            self.roc_auc = 0.0
        self.recall = recall_score(self._ytrue, ypred, labels=[0, 1])
        self.precision = precision_score(self._ytrue, ypred, labels=[0, 1])
        self.average_precision = average_precision_score(self._ytrue, ypred)

    def update(self, other: "TrainMetric"):
        """Method to update such a class with another given one"""
        i = self._threshold[-1][1] if self._threshold else 0
        for (t, j) in other._threshold:
            self._threshold.append((t, i+j))
        self._ypred = np.concatenate(
            (self._ypred, other._ypred), axis=0, dtype=np.float64)
        self._ytrue = np.concatenate(
            (self._ytrue, other._ytrue), axis=0, dtype=np.float64)
        self.__post_init__()

    @classmethod
    def get_metrics(cls):
        """Method to return all the metrics defined in this class"""
        return [k.name for k in fields(cls) if k.repr and k.name not in ("tp", "tn", "fp", "fn")]


@dataclass
class BaseStats(UpdatableDataclass):
    """Base statistic class holding stats concerning a certain type of flows (new, etc.)"""
    sessions: int = 0
    mitigated_sessions: int = 0

    metered_packets: int = 0
    unmetered_packets: int = 0
    sessions_with_unmetered_packets: int = 0

    mitigator_memory: int = 0
    traffic_analyser_memory: int = 0
    detection_engine_memory: int = 0
    traffic_analyser_cpu: int = 0
    detection_engine_cpu: int = 0


@dataclass
class Stats(UpdatableDataclass):
    """Stats class to hold information about all kind of flows within a certain
    type (TP, FP)."""
    # newly monitored
    new: BaseStats = field(default_factory=BaseStats)
    # already monitored and with the same prediction
    from_same: BaseStats = field(default_factory=BaseStats)
    # already monitored but with the opposite prediction
    from_opposite: BaseStats = field(default_factory=BaseStats)

    ignored_sessions: int = 0
    ignored_packets: int = 0

    mitigated_packets: int = 0
    mitigated_sessions_reappeared: int = 0
    mitigator_cpu: int = 0

    def __post_init__(self):
        for k in ("new", "from_same", "from_opposite"):
            if isinstance(getattr(self, k), dict):
                setattr(self, k, BaseStats(**getattr(self, k)))


@dataclass
class Times(UpdatableDataclass):
    """Class to hold times recorded during the online test"""
    blacklist_time: float = 0
    extraction_time: float = 0
    preprocessing_time: float = 0
    conversion_time: float = 0
    predict_time: float = 0

    @property
    def no_conversion(self):
        return self.blacklist_time + self.extraction_time + self.preprocessing_time + self.predict_time

    @property
    def total(self):
        return self.no_conversion + self.conversion_time


@dataclass
class TestMetric(TrainMetric):
    """Class to hold all metrics used within the online test. Additional information
    concerning each nature of flows (TP, etc.) are mantained separetely."""
    flows_per_second: float = 0.0
    packets_per_second: float = 0.0
    packets_fnr_time_window: float = 0.0
    packets_fnr_early_mitigation: float = 0.0
    packets_tpr_time_window: float = 0.0
    packets_tpr_early_mitigation: float = 0.0
    packets_tnr_time_window: float = 0.0
    packets_tnr_early_mitigation: float = 0.0
    packets_fpr_time_window: float = 0.0
    packets_fpr_early_mitigation: float = 0.0
    benign_traffic_metered_percentage: float = 0.0
    malicious_traffic_metered_percentage: float = 0.0
    traffic_metered_percentage: float = 0.0
    flows_with_unmetered_packets_percentage: float = 0.0
    benign_flows_with_unmetered_packets_percentage: float = 0.0
    malicious_flows_with_unmetered_packets_percentage: float = 0.0
    estimated_traffic_analyser_memory_bytes: int = 0
    estimated_traffic_analyser_cpu_instructions: int = 0
    estimated_detection_engine_memory_bytes: int = 0
    estimated_detection_engine_cpu_instructions: int = 0
    estimated_mitigator_memory_bytes: int = 0
    estimated_mitigator_cpu_instructions: int = 0
    estimated_memory_bytes: int = 0
    estimated_cpu_instructions: int = 0

    times: Times = field(default_factory=Times)
    tp_stats: Stats = field(default_factory=Stats)
    fp_stats: Stats = field(default_factory=Stats)
    tn_stats: Stats = field(default_factory=Stats)
    fn_stats: Stats = field(default_factory=Stats)

    def __post_init__(self):
        """If the class is not empty, then compute all the metrics"""
        super().__post_init__()
        for k in ("tn_stats", "fn_stats", "tp_stats", "fp_stats"):
            if not isinstance(getattr(self, k), Stats):
                setattr(self, k, Stats(**getattr(self, k)))
        if not isinstance(self.times, Times):
            self.times = Times(**self.times)
        if self._ypred.size == 0:
            return
        self.flows_with_unmetered_packets_percentage = safe_division(self.get_stats_for(
            "sessions_with_unmetered_packets"), self.tp + self.tn + self.fp + self.fn)
        self.benign_flows_with_unmetered_packets_percentage = safe_division(self.get_stats_for(
            "sessions_with_unmetered_packets", ("tn_stats", "fp_stats")), self.tn + self.fp)
        self.malicious_flows_with_unmetered_packets_percentage = safe_division(self.get_stats_for(
            "sessions_with_unmetered_packets", ("tp_stats", "fn_stats")), self.tp + self.fn)
        self.estimated_traffic_analyser_memory_bytes = self.get_stats_for(
            "traffic_analyser_memory")
        self.estimated_detection_engine_memory_bytes = self.get_stats_for(
            "detection_engine_memory")
        self.estimated_mitigator_memory_bytes = self.get_stats_for(
            "mitigator_memory")
        self.estimated_memory_bytes = self.estimated_traffic_analyser_memory_bytes + \
            self.estimated_mitigator_memory_bytes + \
            self.estimated_detection_engine_memory_bytes
        self.estimated_mitigator_cpu_instructions = self.get_stats_for(
            "mitigator_cpu")
        self.estimated_traffic_analyser_cpu_instructions = self.get_stats_for(
            "traffic_analyser_cpu")
        self.estimated_detection_engine_cpu_instructions = self.get_stats_for(
            "detection_engine_cpu")
        self.estimated_cpu_instructions = self.estimated_mitigator_cpu_instructions + \
            self.estimated_traffic_analyser_cpu_instructions + \
            self.estimated_detection_engine_cpu_instructions
        self.flows_per_second = safe_division(
            (self.tp + self.tn + self.fp + self.fn) * 10**9, self.times.no_conversion)
        self.packets_per_second = safe_division(
            self.total_packets * 10**9, self.times.no_conversion)
        self.packets_fnr_time_window = safe_division(self.get_stats_for(
            ("metered_packets", "unmetered_packets"), ("tp_stats", "fn_stats")), self.total_malicious_packets)
        self.packets_fnr_early_mitigation = safe_division(self.get_stats_for(
            "metered_packets", "tp_stats") + self.get_stats_for(("metered_packets", "unmetered_packets"), "fn_stats"),
            self.total_malicious_packets)
        self.packets_tpr_time_window = 1 - self.packets_fnr_time_window
        self.packets_tpr_early_mitigation = 1 - self.packets_fnr_early_mitigation
        self.packets_tnr_time_window = safe_division(self.get_stats_for(
            ("metered_packets", "unmetered_packets"), ("tn_stats", "fp_stats")), self.total_benign_packets)
        self.packets_tnr_early_mitigation = safe_division(self.get_stats_for(
            "metered_packets", "fp_stats") + self.get_stats_for(("metered_packets", "unmetered_packets"), "tn_stats"),
            self.total_benign_packets)
        self.packets_fpr_time_window = 1 - self.packets_tnr_time_window
        self.packets_fpr_early_mitigation = 1 - self.packets_tnr_early_mitigation
        self.benign_traffic_metered_percentage = safe_division(self.get_stats_for(
            "metered_packets", ("tn_stats", "fp_stats")), self.total_benign_packets)
        self.malicious_traffic_metered_percentage = safe_division(self.get_stats_for(
            "metered_packets", ("tp_stats", "fn_stats")), self.total_malicious_packets)
        self.traffic_metered_percentage = safe_division(
            self.get_stats_for("metered_packets"), self.total_packets)

    def get_stats_for(
            self,
            attrs: Union[str, Tuple[str]],
            natures: Union[str, Tuple[str]] = (
                "tp_stats", "fp_stats", "tn_stats", "fn_stats"),
            types: Union[str, Tuple[str]] = ("new", "from_same", "from_opposite")):
        """Methods to return the desired values from all attributes matching the provided ones"""
        if not isinstance(attrs, (tuple, list)):
            attrs = (attrs, )
        if not isinstance(natures, (tuple, list)):
            natures = (natures, )
        if not isinstance(types, (tuple, list)):
            types = (types, )

        third_nested = tuple(
            x for x in attrs if next((y for y in fields(BaseStats) if x == y.name), False))
        second_nested = tuple(
            x for x in attrs if next((y for y in fields(Stats) if x == y.name), False))

        return sum(getattr(getattr(getattr(self, x), y), z) for x in natures for y in types for z in third_nested) +\
            sum(getattr(getattr(self, x), z)
                for x in natures for z in second_nested)

    @classmethod
    def get_metrics(cls):
        """Method to return all the metrics defined by this class"""
        return [k.name for k in fields(cls) if k.repr and k.name not in (
            "tp", "tn", "fp", "fn", "tp_stats", "tn_stats", "fp_stats", "fn_stats", "times")]

    @staticmethod
    def get_type_from_pred_true(is_predicted_malicious: bool, is_malicious: bool) -> str:
        """Method to return the nature of a flow given its true value and the predicted one"""
        if is_malicious and is_predicted_malicious:
            return "tp"
        elif is_predicted_malicious and is_malicious != is_predicted_malicious:
            return "fp"
        elif not is_predicted_malicious and not is_malicious:
            return "tn"
        elif not is_predicted_malicious and is_malicious != is_predicted_malicious:
            return "fn"
        else:
            raise ValueError("Do not know how to infer")

    def update(self, other: "TestMetric"):
        """Method to update this instance with another one"""
        [getattr(self, x).update(getattr(other, x))
         for x in ("times", "tp_stats", "fp_stats", "tn_stats", "fn_stats")]
        super().update(other)

    @property
    def total_malicious_packets(self):
        return self.get_stats_for(("metered_packets", "unmetered_packets", "mitigated_packets"),
                                  ("tp_stats", "fn_stats"))

    @property
    def total_benign_packets(self):
        return self.get_stats_for(("metered_packets", "unmetered_packets", "mitigated_packets", "ignored_packets"),
                                  ("tn_stats", "fp_stats"))

    @property
    def total_packets(self):
        return self.get_stats_for(("metered_packets", "unmetered_packets", "mitigated_packets", "ignored_packets"))


@dataclass
class ResultTestCategory(TestMetric):
    captures: Dict[str, TestMetric] = field(default_factory=dict)

    def __post_init__(self):
        super().__post_init__()
        for k in self.captures:
            if not isinstance(self.captures[k], TestMetric):
                self.captures[k] = TestMetric(**self.captures[k])


@dataclass
class ResultTestDataset(TestMetric):
    categories: Dict[str, ResultTestCategory] = field(default_factory=dict)

    def __post_init__(self):
        super().__post_init__()
        for k in self.categories:
            if not isinstance(self.categories[k], ResultTestCategory):
                self.categories[k] = ResultTestCategory(**self.datasets[k])


@dataclass
class ResultTest(TestMetric):
    """Class holding results of the online test divided by granularity."""
    datasets: Dict[str, ResultTestDataset] = field(default_factory=dict)

    def __post_init__(self):
        super().__post_init__()
        for k in self.datasets:
            if not isinstance(self.datasets[k], ResultTestDataset):
                self.datasets[k] = ResultTestDataset(**self.datasets[k])

    def update(self, other: Type["ResultTest"] = None):
        """Method to update such a class with the provided one. If the other one is not provided, then
        this method recomputes the data of this class by starting from the most inner ones (pcap granularity)
        untill the more generic ones are created."""
        if other:
            # for each pcap in other, update pcap of this class.
            # note that we update only pcap, as the category/dataset/general data
            # is updated by invoking this method with no argument
            for k, v in other.datasets.items():
                if k not in self.datasets:
                    self.datasets[k] = ResultTestDataset()
                for kk, vv in v.categories.items():
                    if kk not in self.datasets[k].categories:
                        self.datasets[k].categories[kk] = ResultTestCategory()
                    for kkk, vvv in vv.captures.items():
                        if kkk not in self.datasets[k].categories[kk].captures:
                            self.datasets[k].categories[kk].captures[kkk] = TestMetric(
                            )
                        tresh = vvv._threshold[-1][0]
                        self.datasets[k].categories[kk].captures[kkk]._ypred = np.concatenate(
                            (self.datasets[k].categories[kk].captures[kkk]._ypred, vvv._ypred), axis=0, dtype=np.float64)
                        self.datasets[k].categories[kk].captures[kkk]._ytrue = np.concatenate(
                            (self.datasets[k].categories[kk].captures[kkk]._ytrue, vvv._ytrue), axis=0, dtype=np.float64)
                        self.datasets[k].categories[kk].captures[kkk]._threshold.append(
                            (tresh, len(self.datasets[k].categories[kk].captures[kkk]._ytrue)))
                        [getattr(self.datasets[k].categories[kk].captures[kkk], x).update(getattr(
                            vvv, x)) for x in ("times", "tp_stats", "fp_stats", "tn_stats", "fn_stats")]
            for k, v in self.datasets.items():
                for kk, vv in v.categories.items():
                    for kkk, vvv in vv.captures.items():
                        if k not in other.datasets or kk not in other.datasets[k].categories or\
                                kkk not in other.datasets[k].categories[kk].captures:
                            self.datasets[k].categories[kk].captures[kkk]._threshold.append(
                                (tresh, len(self.datasets[k].categories[kk].captures[kkk]._ytrue)))
        else:
            # Recompute stats starting by the pcap granularity until the more generic one
            self._ypred = np.array([], dtype=np.float64)
            self._ytrue = np.array([], dtype=np.float64)
            self.times = Times()
            [setattr(self, x, Stats())
             for x in ("tp_stats", "fp_stats", "tn_stats", "fn_stats")]

            for v in self.datasets.values():
                v._ypred = np.array([], dtype=np.float64)
                v._ytrue = np.array([], dtype=np.float64)
                v.times = Times()
                [setattr(v, x, Stats())
                 for x in ("tp_stats", "fp_stats", "tn_stats", "fn_stats")]

                for vv in v.categories.values():
                    vv._ypred = np.array([], dtype=np.float64)
                    vv._ytrue = np.array([], dtype=np.float64)
                    vv.times = Times()
                    [setattr(vv, x, Stats()) for x in (
                        "tp_stats", "fp_stats", "tn_stats", "fn_stats")]

                    for vvv in vv.captures.values():
                        vvv.__post_init__()
                        vv._ypred = np.concatenate(
                            (vv._ypred, vvv._ypred), axis=0, dtype=np.float64)
                        vv._ytrue = np.concatenate(
                            (vv._ytrue, vvv._ytrue), axis=0, dtype=np.float64)
                        [getattr(vv, x).update(getattr(vvv, x)) for x in (
                            "times", "tp_stats", "fp_stats", "tn_stats", "fn_stats")]
                    it = next(vvv._threshold for vvv in vv.captures.values())
                    vv._threshold = [(it[i][0], sum(
                        vvv._threshold[i][1] for vvv in vv.captures.values())) for i in range(len(it))]
                    vv.__post_init__()

                    v._ypred = np.concatenate(
                        (v._ypred, vv._ypred), axis=0, dtype=np.float64)
                    v._ytrue = np.concatenate(
                        (v._ytrue, vv._ytrue), axis=0, dtype=np.float64)
                    [getattr(v, x).update(getattr(vv, x)) for x in (
                        "times", "tp_stats", "fp_stats", "tn_stats", "fn_stats")]

                it = next(vv._threshold for vv in v.categories.values())
                v._threshold = [(it[i][0], sum(vv._threshold[i][1]
                                 for vv in v.categories.values())) for i in range(len(it))]
                v.__post_init__()

                self._ypred = np.concatenate(
                    (self._ypred, v._ypred), axis=0, dtype=np.float64)
                self._ytrue = np.concatenate(
                    (self._ytrue, v._ytrue), axis=0, dtype=np.float64)
                [getattr(self, x).update(getattr(v, x)) for x in (
                    "times", "tp_stats", "fp_stats", "tn_stats", "fn_stats")]
            it = next(v._threshold for v in self.datasets.values())
            self._threshold = [(it[i][0], sum(v._threshold[i][1]
                                for v in self.datasets.values())) for i in range(len(it))]
            self.__post_init__()
