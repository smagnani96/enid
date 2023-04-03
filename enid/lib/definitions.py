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
"""File containing the most important definitions of classes and behaviours."""
import importlib
import os
import time
from itertools import cycle
from abc import ABC, abstractclassmethod, abstractmethod
from dataclasses import dataclass, field, replace, fields
from enum import Enum
from typing import Dict, List, Tuple, Type, Union, OrderedDict

import joblib
import numpy as np
from pypacker.layer3.ip import IP
from pypacker.ppcap import Reader

from .identifiers import BaseKey, TwoIPsProtoPortsKey, str_to_key
from .metrics import BaseStats, ComputationalRequirements, Stats, TestMetric, ResultTest, ResultTestDataset, ResultTestCategory
from .utility import (CDataJSONEncoder, EthInPcap, camel_to_snake, get_logger, safe_division,
                      snake_to_camel, get_all_dict_comb, load_json_data)


@dataclass
class BaseFeature(ABC):
    """BaseFeature class. All instances must comply to this interface."""
    value: int = 0

    @property
    @abstractmethod
    def computational_requirements(cls) -> Tuple[ComputationalRequirements]:
        """Method to return a list or tuple of computational requirements."""
        pass

    @property
    @abstractmethod
    def memory_requirements(cls) -> int:
        """Method to return the memory bytes corresponding to the feature"""
        pass

    @classmethod
    def create(cls, eth: EthInPcap):
        """Method to create a new instance of the feature from the packet"""
        ret = cls()
        ret.extract(eth)
        return ret

    @abstractmethod
    def extract(self, eth: EthInPcap):
        """Method to extract/update the current instance of the feature from the packet"""
        pass

    def to_json(self):
        """Method to return the feature in a json-like format"""
        return self.value


@dataclass
class AggBaseFeature(BaseFeature):
    """AggbaseFeature class that represents the base class for all features that are
    statistics instead of single values, hence updated throughout the time."""
    @classmethod
    def create(cls, eth: EthInPcap, is_fwd=False):
        ret = cls()
        ret.extract(eth, is_fwd)
        return ret

    @abstractmethod
    def extract(self, eth: EthInPcap, is_fwd=False):
        pass


@dataclass
class SessionValue:
    """Class to hold the values monitored from a flow"""
    prediction: float = 0
    metered_packets: int = 0
    unmetered_packets: int = 0
    value: Union[Tuple[AggBaseFeature],
                 List[Tuple[BaseFeature]]] = field(default_factory=list)

    @property
    def total_packets(self):
        return self.metered_packets + self.unmetered_packets


class Ticker:
    """Class to represent a ticker that hold a list of tasks and everytime the tick method
    is called their age is reduced until zero."""

    def __init__(self, missing_tw, to_blacklist) -> None:
        self.missing_tw: int = missing_tw
        self.to_blacklist: List[Type[BaseKey]] = to_blacklist

    def tick(self):
        self.missing_tw -= 1

    @property
    def ready(self):
        return self.missing_tw < 0


@dataclass
class FeaturesHolder(ABC):
    """Abstract class for holding a features and defining which ones are
    allowed within a Detection Model."""
    value: OrderedDict[Type[BaseFeature], Union[float,
                                                Tuple[float, float]]] = field(default=None)

    def __iter__(self):
        for x in range(len(self.ALLOWED), 0, -1):
            yield x

    @classmethod
    @property
    @abstractmethod
    def ALLOWED(cls) -> Tuple[Type[BaseFeature]]:
        """All the allowed ones"""
        pass

    @property
    def computational_requirements(self):
        """Method to return the cpu requirements of extracting the current ones"""
        return tuple(k for x in self.value for k in x.computational_requirements)

    def get_feature_value(self, y: BaseFeature):
        return self.value[y] if y in self.value else None

    @property
    def memory_requirements(self):
        """Method to return the memory requirements of extracting the current ones"""
        return sum(x.memory_requirements for x in self.value)

    @abstractmethod
    def pop_less_relevant(self, key_depth_class: Type[BaseKey] = None) -> Type[BaseFeature]:
        """Abstract method to remove the least important features from the
        currently active ones"""
        pass

    @property
    def n_total(self) -> int:
        """All the allowed ones"""
        return len(self.ALLOWED)

    @property
    def n_current(self) -> int:
        """Only the current ones"""
        return len(self.value)


@dataclass
class DeParams:
    """Base class to hold the parameters of a Detection Model"""
    packets_per_session: int = None
    features: int = None
    max_packets_per_session: int = None
    max_features: int = None
    malicious_threshold: int = 0
    key_depth_class: Type[BaseKey] = TwoIPsProtoPortsKey

    def to_json(self):
        """Method to dump the class in a json-style"""
        e = CDataJSONEncoder()
        return {k.name: e.default(getattr(self, k.name)) for k in fields(self) if k.repr}

    def __post_init__(self):
        if isinstance(self.key_depth_class, str):
            self.key_depth_class = str_to_key(self.key_depth_class)

    def all_train_combs(self, exclude=None):
        """Method to return all combination of trainable models"""
        return get_all_dict_comb({k: getattr(self, k) for k in self.train_combs() if k != exclude})

    def previous_one(self, pps, ftrs):
        """Method to return the previous model parameters and the 'distance'
        with the current one"""
        is_max_features = ftrs == max(self.features)
        if pps is None:
            if is_max_features:
                return False
            prev_features = self.features[self.features.index(ftrs)-1]
            return pps, prev_features, prev_features - ftrs, 0

        is_max_packets = pps == max(self.packets_per_session)
        if is_max_features and is_max_packets:
            return False
        if is_max_packets:
            prev_features = self.features[self.features.index(ftrs)-1]
            return pps, prev_features, prev_features - ftrs, 0
        return self.packets_per_session[self.packets_per_session.index(pps)-1], ftrs, 0

    def train_combs(self):
        """Method for returning the combination of parameters for generating models"""
        return tuple(x for x in ("packets_per_session", "features") if getattr(self, x) is not None)


@dataclass
class AnalysisState:
    """Base class to represent the state of an analysis"""
    time_window: int = None
    sessions_per_timewindow: int = None
    enforce_timewindows_delay: int = 0
    is_adaptiveness_enabled: bool = False

    current_key: Type[BaseKey] = field(default=None)
    current_features: Type[FeaturesHolder] = field(default=None)
    params: DeParams = field(default=None)

    def __post_init__(self):
        if isinstance(self.current_key, str):
            self.current_key = str_to_key(self.current_key)
        if isinstance(self.current_features, dict):
            self.current_features = self.__class__.current_features.__class__(
                **self.current_features)
        if isinstance(self.params, dict):
            self.params = self.__class__.params.__class__(**self.params)


class TestType(Enum):
    """Enumeration to hold the type of the online test"""
    PROCESSING = 0
    THROUGHPUT = 1
    NORMAL = 2


class DebugLevel(Enum):
    """Enumeration to hold the debug level of the online test"""
    NONE = 0
    BASE = 1
    ENHANCED = 2


@dataclass
class BaseProcessingData:
    """Base data that need to be generated from a processing method while parsing pcap
    with the preprocesser.py program"""
    tot_benign: int = 0
    tot_malicious: int = 0
    unique_benign: int = 0
    unique_malicious: int = 0
    tot_benign_packets: int = 0
    tot_malicious_packets: int = 0


class TrafficAnalyser(ABC):
    """Traffic Analyser that includes the Filtering mechanism to drop packets matching
    the blacklist and the Feature Extraction mechanisms. In addition to that, it contains a
    reference to the Detection Model of interests and all the parameters used."""
    analysis_state_cls: Type[AnalysisState] = AnalysisState
    session_value_cls: Type[SessionValue] = SessionValue

    def __init__(self, detection_engine_cls: Type["DetectionEngine"],
                 attackers: Dict[str, Tuple[Type[BaseKey]]],
                 time_window: int, current_key: Type[BaseKey], test_type: TestType,
                 models_dir: str = None, dump_path: str = None,
                 enforce_timewindows_delay: int = None,
                 is_adaptiveness_enabled: bool = None, sessions_per_timewindow=None,
                 debug: DebugLevel = DebugLevel.NONE, **kwargs):

        if test_type == TestType.PROCESSING:
            params = detection_engine_cls.de_params(
                packets_per_session=kwargs.get("packets_per_session", None),
                features=len(detection_engine_cls.features_holder_cls.ALLOWED))
            hold = detection_engine_cls.features_holder_cls()
        else:
            name = detection_engine_cls.model_name(**kwargs)
            hold = detection_engine_cls.features_holder_cls(**load_json_data(
                os.path.join(models_dir, name, "relevance.json")))
            params = detection_engine_cls.de_params(**load_json_data(
                os.path.join(models_dir, name, "params.json")))

        self.analysis_state = self.analysis_state_cls(
            time_window=time_window,
            sessions_per_timewindow=sessions_per_timewindow,
            enforce_timewindows_delay=enforce_timewindows_delay,
            is_adaptiveness_enabled=is_adaptiveness_enabled,
            current_key=current_key,
            current_features=hold,
            params=params)

        self.test_type = test_type
        self.dump_path = dump_path
        self.start_time_window: int = None

        self.blacklist_times = {}
        self.extraction_times = {}

        self.black_map: Dict[Type[BaseKey], bool] = {}
        self.seen_sessions_previous_prediction: Dict[Type[BaseKey], str] = {}

        self.current_untracked_map: Dict[Type[BaseKey], int] = {}
        self.current_black_map: Dict[Type[BaseKey], int] = {}
        self.current_session_map: Dict[Type[BaseKey], Type[SessionValue]] = {}

        self.enforce_tasks: List[Ticker] = []
        self.de: Type[DetectionEngine] = detection_engine_cls(
            self.analysis_state, models_dir)
        if test_type != TestType.PROCESSING:
            self.de.load_model()
        self.results = ResultTest()
        self.attackers = {ds: (
            next(x for x in attackers if x.dataset == ds).__class__,
            {x: None for x in attackers if x.dataset == ds}) for ds in set([x.dataset for x in attackers])}
        self.n_timewindow = 0
        if test_type == TestType.PROCESSING:
            self.processing_stats = detection_engine_cls.processing_data_cls()
        self.debug = debug
        if debug != DebugLevel.NONE:
            self.debug_data = {}

    @abstractmethod
    def _extract(self, sess_id: Type[BaseKey], eth: EthInPcap):
        """Method defined by the TrafficAnalyser of each Detection Model for
        the feature extraction of the sessions under monitoring."""
        pass

    def _terminate_timewindow(self):
        """Method invoked at the end of each time window. Here the data is used
        for the classification, and the malicious sessions are blacklisted accordingly.
        In addition, the method to compute costs and statistics is called."""
        self.n_timewindow += 1
        de_cpu, conversion_time, preprocessing_time, predict_time, de_mem, y_pred = 0, 0, 0, 0, 0, None
        # check if at least 1 session is monitored.
        if self.current_session_map:
            de_mem, y_pred, conversion_time, preprocessing_time, predict_time = self.de.predict(
                self)
            de_cpu = self.de.parameters(model=self.de.model)
            tmp = []
            # assign the prediction to each flow and create the list of flows to be blacklisted
            for p, (sess_id, v) in zip(y_pred, self.current_session_map.items()):
                v.prediction = p.item()
                if v.prediction > self.analysis_state.params.malicious_threshold and sess_id not in self.black_map:
                    tmp.append(sess_id)
            if tmp:
                # insert the list as a Ticker task
                self.enforce_tasks.append(
                    Ticker(self.analysis_state.enforce_timewindows_delay, tmp))

        if self.enforce_tasks:
            # decrease all tasks by 1 and apply rules of all the ready ones
            [x.tick() for x in self.enforce_tasks]
            if self.enforce_tasks[0].ready:
                t = self.enforce_tasks.pop(0)
                for x in t.to_blacklist:
                    if x not in self.black_map:
                        self.black_map[x] = True

        # compute costs and statistics for both the current time window and the global results
        self._compute_cost(de_cpu, de_mem, self.blacklist_times, self.extraction_times,
                           conversion_time, preprocessing_time, predict_time)

        if self.analysis_state.is_adaptiveness_enabled:
            raise NotImplementedError("To be implemented")

    @abstractmethod
    def _terminate_timewindow_preprocessing(self):
        """Method defined by the TrafficAnalyser of each Detection Model to be
        invoked at the end of a monitoring time window while preprocessing data."""
        raise NotImplementedError()

    def _new_packet_preprocesser(self, ts, eth: EthInPcap = None):
        """Method used to handle a packet while preprocessing data"""
        # set current start of the window if not already set
        if not self.start_time_window:
            self.start_time_window = ts

        # check whether time window is finished
        if ts - self.start_time_window >= self.analysis_state.time_window:
            # invoke method and clear all data structures (except global blacklist)
            self._terminate_timewindow_preprocessing()
            self.current_session_map.clear()
            self.start_time_window = None

        # check if IP packet or valid ethernet buffer
        if eth is None or not eth[IP]:
            return None

        # compute the session identifier
        sess_id = self.analysis_state.current_key.extract(eth)
        # check if it is possible to monitor the session
        if sess_id not in self.current_session_map:
            self.current_session_map[sess_id] = self.session_value_cls()

        # check if session already reached the max number of monitored packets
        if self.analysis_state.params.packets_per_session and\
                self.current_session_map[sess_id].metered_packets == self.analysis_state.params.packets_per_session:
            self.current_session_map[sess_id].unmetered_packets += 1
            return

        self.current_session_map[sess_id].metered_packets += 1

        # compute and execute the feature extraction
        self._extract(sess_id, eth)

    def _new_packet(self, ts, eth: EthInPcap = None):
        """Method used to handle a new ethernet packet"""
        # set current start of the window if not already set
        if not self.start_time_window:
            self.start_time_window = ts

        # check whether time window is finished
        if ts - self.start_time_window >= self.analysis_state.time_window:
            # invoke method and clear all data structures (except global blacklist)
            self._terminate_timewindow()
            self.current_session_map.clear()
            self.current_black_map.clear()
            self.current_untracked_map.clear()
            self.extraction_times = {}
            self.blacklist_times = {}
            self.start_time_window = None

        # check if IP packet or valid ethernet buffer
        if eth is None or not eth[IP]:
            return None

        # compute the session identifier
        sess_id = self.analysis_state.current_key.extract(eth)

        # compute time for the blacklist lookup
        t = time.clock_gettime_ns(time.CLOCK_PROCESS_CPUTIME_ID)
        is_blacklisted = sess_id in self.black_map
        t = time.clock_gettime_ns(time.CLOCK_PROCESS_CPUTIME_ID) - t
        if (sess_id.dataset, sess_id.category, sess_id.pcap) not in self.blacklist_times:
            self.blacklist_times[(
                sess_id.dataset, sess_id.category, sess_id.pcap)] = 0
        self.blacklist_times[(
            sess_id.dataset, sess_id.category, sess_id.pcap)] += t

        # block packet if blacklisted and NORMAL test
        if not self.test_type == TestType.THROUGHPUT and is_blacklisted:
            if sess_id not in self.current_black_map:
                self.current_black_map[sess_id] = 0
            self.current_black_map[sess_id] += 1
            return

        #  skip the analysis of the packet if the session is not monitored
        if sess_id in self.current_untracked_map:
            self.current_untracked_map[sess_id] += 1
            return

        # check if it is possible to monitor the session
        if sess_id not in self.current_session_map:
            # check if enough space
            if self.analysis_state.sessions_per_timewindow and\
                    len(self.current_session_map) == self.analysis_state.sessions_per_timewindow:
                self.current_untracked_map[sess_id] = 1
                return
            self.current_session_map[sess_id] = self.session_value_cls()

        # check if session already reached the max number of monitored packets
        if self.analysis_state.params.packets_per_session and\
                self.current_session_map[sess_id].metered_packets == self.analysis_state.params.packets_per_session:
            self.current_session_map[sess_id].unmetered_packets += 1
            return

        self.current_session_map[sess_id].metered_packets += 1

        # compute and execute the feature extraction
        t = time.clock_gettime_ns(time.CLOCK_PROCESS_CPUTIME_ID)
        self._extract(sess_id, eth)
        t = time.clock_gettime_ns(time.CLOCK_PROCESS_CPUTIME_ID) - t
        if (sess_id.dataset, sess_id.category, sess_id.pcap) not in self.extraction_times:
            self.extraction_times[(
                sess_id.dataset, sess_id.category, sess_id.pcap)] = 0
        self.extraction_times[(
            sess_id.dataset, sess_id.category, sess_id.pcap)] += t

    @staticmethod
    def generate_packets(pcap, analyser: "TrafficAnalyser", identifier=None, labels=None):
        """Method to generate packets  for the analyser provided from the pcap.
        If present, this function tries to load the packets' labels, such as the
        dataset category and pcap of belonging."""
        if identifier is None:
            identifier = analyser.de.model_name(
                **analyser.analysis_state.params.__dict__)
        logger = get_logger(identifier)
        if not labels:
            with open(pcap.replace(".pcap", ".joblib"), "rb") as fp:
                labels = joblib.load(fp)
            method = analyser._new_packet
        else:
            labels = cycle([labels])
            method = analyser._new_packet_preprocesser
        tot_bytes = os.path.getsize(pcap)
        curr_bytes = 0

        for curr_pkts, ((s_dataset, s_category, s_pcap), (ts, buf)) in enumerate(zip(labels, Reader(filename=pcap))):
            curr_bytes += len(buf) + 16  # timestamp in nanoseconds
            if curr_pkts % 50000 == 0:
                logger.info("Read {}% bytes ({}/{}) and packet n°{}".format(
                    round(curr_bytes*100/tot_bytes, 2), curr_bytes, tot_bytes, curr_pkts))
            try:
                eth = EthInPcap(ts, s_dataset, s_category, s_pcap, buf)
            except Exception:
                eth = None

            method(ts, eth=eth)
        if analyser.test_type == TestType.PROCESSING:
            analyser._terminate_timewindow_preprocessing()
        else:
            analyser._terminate_timewindow()
            analyser.results.update()

        logger.info("Finished")

    def _get_sess_type(self, sess_id):
        """Method to return 0/1 whether the session identifier is malicious"""
        return int(sess_id.cast(self.attackers[sess_id.dataset][0]) in self.attackers[sess_id.dataset][1])

    def _compute_cost(self, de_cpu, de_mem, blacklist_times, extraction_times, conversion_time,
                      preprocessing_time, predict_time, **kwargs):
        """Method for computing costs and statistics from the results of the current time
        interval. Results are also propagated to the global results of the test accordingly."""
        tw_res = ResultTest()

        # retrieve key and feature computational costs
        key_comp_req = ComputationalRequirements.requirements_to_cost(
            self.analysis_state.current_key.computational_requirements)
        feat_comp_req = ComputationalRequirements.requirements_to_cost(
            self.analysis_state.current_features.computational_requirements, ignore_depth=self.analysis_state.current_key)

        if self.current_session_map:
            pcap = {}
            n_samples = len(self.current_session_map)
            # adjust times to refer to a single flow
            predict_time = safe_division(predict_time, n_samples, default=0.0)
            preprocessing_time = safe_division(
                preprocessing_time, n_samples, default=0.0)
            conversion_time = safe_division(
                conversion_time, n_samples, default=0.0)
            for sess_id, v in self.current_session_map.items():
                is_malicious = self._get_sess_type(sess_id)
                prev_prediction = self.seen_sessions_previous_prediction.get(
                    sess_id, None)
                is_predicted_malicious = v.prediction > self.analysis_state.params.malicious_threshold

                ttype = TestMetric.get_type_from_pred_true(
                    is_predicted_malicious, is_malicious)

                # check if dataset, category and pcap already present in results
                if sess_id.dataset not in tw_res.datasets:
                    tw_res.datasets[sess_id.dataset] = ResultTestDataset()
                if sess_id.category not in tw_res.datasets[sess_id.dataset].categories:
                    tw_res.datasets[sess_id.dataset].categories[sess_id.category] = ResultTestCategory(
                    )
                if sess_id.pcap not in tw_res.datasets[sess_id.dataset].categories[sess_id.category].captures:
                    tw_res.datasets[sess_id.dataset].categories[sess_id.category].captures[sess_id.pcap] = TestMetric(
                    )
                if (sess_id.dataset, sess_id.category, sess_id.pcap) not in pcap:
                    pcap[(sess_id.dataset, sess_id.category, sess_id.pcap)] = (
                        [], [])
                # appending ytrue and ypred
                pcap[(sess_id.dataset, sess_id.category, sess_id.pcap)
                     ][0].append(is_malicious)
                pcap[(sess_id.dataset, sess_id.category, sess_id.pcap)
                     ][1].append(v.prediction)

                # update times
                t = tw_res.datasets[sess_id.dataset].categories[sess_id.category].captures[sess_id.pcap]
                t.times.conversion_time += conversion_time
                t.times.predict_time += predict_time
                t.times.preprocessing_time += preprocessing_time

                # depending by the nature of the session (TP, FN, etc.) update metrics accordingly
                t1: BaseStats = getattr(
                    getattr(
                        tw_res.datasets[sess_id.dataset].categories[sess_id.category].captures[sess_id.pcap],
                        f"{ttype}_stats"),
                    "new" if prev_prediction is None else
                    "from_same" if prev_prediction == ttype else "from_opposite")
                t1.traffic_analyser_memory += self.analysis_state.current_key.memory_requirements + \
                    self.analysis_state.current_features.memory_requirements * \
                    (1 if isinstance(v.value[0],
                     AggBaseFeature) else v.metered_packets)
                t1.detection_engine_memory += de_mem
                t1.detection_engine_cpu += de_cpu
                t1.traffic_analyser_cpu += key_comp_req * (v.metered_packets+v.unmetered_packets) +\
                    feat_comp_req * v.metered_packets
                t1.sessions += 1

                t1.metered_packets += v.metered_packets
                t1.unmetered_packets += v.unmetered_packets
                if v.unmetered_packets:
                    t1.sessions_with_unmetered_packets += 1
                if is_predicted_malicious and prev_prediction not in ('tp', 'fp'):
                    t1.mitigated_sessions += 1
                    t1.mitigator_memory += self.analysis_state.current_key.memory_requirements
                self.seen_sessions_previous_prediction[sess_id] = ttype

            # Update extraction times for the right pcap
            for (d, c, p), v in extraction_times.items():
                tw_res.datasets[d].categories[c].captures[p].times.extraction_time += v

            # update ytrue, ypred and threshold accordingly
            for (d, c, p), v in pcap.items():
                tw_res.datasets[d].categories[c].captures[p]._ytrue = np.array(
                    v[0], dtype=np.float64)
                tw_res.datasets[d].categories[c].captures[p]._ypred = np.array(
                    v[1], dtype=np.float64)
                tw_res.datasets[d].categories[c].captures[p]._threshold = [
                    (self.analysis_state.params.malicious_threshold, len(v[0]))]

        # for each untracked session update stats
        for sess_id, v in self.current_untracked_map.items():
            if sess_id.dataset not in tw_res.datasets:
                tw_res.datasets[sess_id.dataset] = ResultTestDataset()
            if sess_id.category not in tw_res.datasets[sess_id.dataset].categories:
                tw_res.datasets[sess_id.dataset].categories[sess_id.category] = ResultTestCategory(
                )
            if sess_id.pcap not in tw_res.datasets[sess_id.dataset].categories[sess_id.category].captures:
                tw_res.datasets[sess_id.dataset].categories[sess_id.category].captures[sess_id.pcap] = TestMetric(
                    _threshold=[(self.analysis_state.params.malicious_threshold, 0)])
            t2: Stats = getattr(tw_res.datasets[sess_id.dataset].categories[sess_id.category].captures[sess_id.pcap],
                                'tp_stats' if self._get_sess_type(sess_id) else 'fp_stats')
            t2.ignored_sessions += 1
            t2.ignored_packets += v

        # foreach session that appeared in the blacklist in the current time interval
        # update stats accordingly
        for sess_id, v in self.current_black_map.items():
            if sess_id.dataset not in tw_res.datasets:
                tw_res.datasets[sess_id.dataset] = ResultTestDataset()
            if sess_id.category not in tw_res.datasets[sess_id.dataset].categories:
                tw_res.datasets[sess_id.dataset].categories[sess_id.category] = ResultTestCategory(
                )
            if sess_id.pcap not in tw_res.datasets[sess_id.dataset].categories[sess_id.category].captures:
                tw_res.datasets[sess_id.dataset].categories[sess_id.category].captures[sess_id.pcap] = TestMetric(
                    _threshold=[(self.analysis_state.params.malicious_threshold, 0)])
            t3: Stats = getattr(tw_res.datasets[sess_id.dataset].categories[sess_id.category].captures[sess_id.pcap],
                                'tp_stats' if self._get_sess_type(sess_id) else 'fp_stats')
            t3.mitigated_packets += v
            t3.mitigated_sessions_reappeared += 1
            t3.mitigator_cpu += key_comp_req * v

        # update blacklist times of the pcap
        for (d, c, p), v in blacklist_times.items():
            tw_res.datasets[d].categories[c].captures[p].times.blacklist_time += v

        # update global results without recomputing metrics (too expensive, done only at the end of the test)
        self.results.update(tw_res)
        # check the debug level and in case recompute metrics and update debug data structures
        if self.debug == DebugLevel.BASE:
            tw_res.update()
            self.debug_data[self.n_timewindow] = replace(tw_res)
        elif self.debug == DebugLevel.ENHANCED:
            tw_res.update()
            self.debug_data[self.n_timewindow] = {
                "result": replace(tw_res),
                "analysis_state": replace(self.analysis_state),
                "session_map": self.current_session_map.copy(),
                "untracked_map": self.current_untracked_map.copy(),
                "black_map": self.current_black_map.copy()
            }


class DetectionEngine(ABC):
    """Main Abstract class for representing a Detection Engine"""

    # class to be used when processing data
    processing_data_cls = BaseProcessingData
    # class to be used for the parameters of the model
    de_params = DeParams

    def __init__(self, analysis_state: Type[AnalysisState], base_dir: str) -> None:
        self.model = None
        self.analysis_state = analysis_state
        self.base_dir: str = base_dir

    def __del__(self):
        del self.model

    @staticmethod
    def list_all(only_main=False):
        """ Method to list all Detection Models available in this framework"""
        ret = []
        basepath = os.path.join(os.path.dirname(__file__), "engines")
        for x in os.listdir(basepath):
            if (only_main and os.path.isdir(os.path.join(
                basepath, x, "main"))) or (not only_main and not x.startswith("_") and
                                           (x.endswith(".py") or os.path.isdir(os.path.join(basepath, x)))):
                ret.append(snake_to_camel(x.replace(".py", "")))
        return ret

    @abstractmethod
    def load_model(self):
        """Abstract method to be implemented, used for loading the model"""
        pass

    @abstractclassmethod
    def parameters(cls, model=None, **kwargs) -> int:
        """Method to be implemented and return the complexity of the model in terms of
        trainable parameters"""
        pass

    @classmethod
    @property
    @abstractmethod
    def features_holder_cls(cls) -> Type[FeaturesHolder]:
        """Feature Holder class to be specified when implementing the Detection Model"""
        pass

    @classmethod
    @property
    @abstractmethod
    def traffic_analyser_cls(cls) -> Type[TrafficAnalyser]:
        """The Traffic Analyser class to be specified when implementing the Detection Model"""
        pass

    @abstractclassmethod
    def _get_arch(*args, **kwargs):
        """Method to get an instance of the model according to the provided arguments"""
        pass

    @abstractmethod
    def preprocess_samples(self, ta: Type[TrafficAnalyser]) -> Tuple[np.ndarray, int, int]:
        """Method to preprocess samples captured by the analyser"""
        pass

    @abstractmethod
    def predict(self, **kwargs) -> Tuple[int, np.ndarray, int, int, int]:
        """Method for classifying the data"""
        pass

    @abstractclassmethod
    def model_name(cls, features: int = None, **kwargs) -> str:
        """Method for returning the name of a model given the parameters provided"""
        pass

    @staticmethod
    def import_de(name: str) -> Type["DetectionEngine"]:
        """Method for importing and returning the Detection Engine class provided as string"""
        return getattr(importlib.import_module('.engines.{}'.format(camel_to_snake(name)), package="enid.lib"), name)

    @abstractclassmethod
    def append_to_dataset(cls, source, dest, ttype, label, indexes=None, **kwargs):
        """Method for appending single processed data to the dataset"""
        pass

    @abstractclassmethod
    def train(cls, dataset_path: str, models_dir: str, params: Type[DeParams], **kwargs):
        """Method for training an instance of the model"""
        pass

    @staticmethod
    def intersect(engines_list: List[Type["DetectionEngine"]]) -> bool:
        """Method to check whether the provided Engines intersect, meaning
        they can be used with the same preprocessed data"""
        engines_list = [x if isinstance(
            x, DetectionEngine) else DetectionEngine.import_de(x) for x in engines_list]
        return all(set(engines_list[0].features_holder_cls.ALLOWED) == set(elem.features_holder_cls.ALLOWED)
                   for elem in engines_list) and\
            all(engines_list[0].append_to_dataset.__code__ ==
                elem.append_to_dataset.__code__ and
                engines_list[0].traffic_analyser_cls._new_packet.__code__ == elem.traffic_analyser_cls._new_packet.__code__
                for elem in engines_list)
