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
"""File defining the LucidCnn Detection Engine, its TrafficAnalyser and all
the parameters used."""
import copy
import os
import time
from dataclasses import dataclass, field, fields, replace
from functools import lru_cache
from typing import List, Tuple, Type

import h5py
import numpy as np
import tensorflow as tf
from pypacker.layer12.ethernet import Ethernet
from sklearn.model_selection import RandomizedSearchCV

from ...definitions import (AnalysisState, BaseProcessingData, DeParams,
                            DetectionEngine, TrafficAnalyser)
from ...identifiers import BaseKey
from ...metrics import TrainMetric
from ...utility import (get_all_dict_comb, get_best_comb,
                        get_best_dispatch, get_logger, load_json_data,
                        set_seed, sort_keep_comb)
from .features import LucidFeaturesHolder, Time

_logger = get_logger(__name__)


@dataclass
class LucidDeParams(DeParams):
    """Class defining parameters for the Detection Model"""
    rerank_at_feature: int = None
    rerank_at_packet: int = None
    train_with_less: bool = False
    prune_zero_first: bool = True
    rank_metric: str = "f1_score"
    max_loss: int = 20
    epochs: int = 500
    malicious_threshold: int = 0.5

    learning_rate: List[float] = (0.1, 0.01, 0.001)
    batch_size: List[int] = (512, 1024, 2048)
    kernels: List[int] = (8, 16, 32, 64)
    dropout: List[float] = (0.2, 0.5, 0.8)
    regularization: List[str] = ("l1", "l2")  # used only for CNN

    @property
    def is_to_train(self):
        """Return true if the current model is to train according to the parameters"""
        return self.train_with_less or (not self.max_packets_per_session and not self.max_features) or\
            (self.max_packets_per_session and self.max_features and self.packets_per_session == self.max_packets_per_session
             and self.features == self.max_features) or\
            (self.max_packets_per_session and not self.max_features and
                self.packets_per_session == self.max_packets_per_session) or\
            (self.max_features and not self.max_packets_per_session and
                self.features == self.max_features)

    @property
    def is_to_rerank(self):
        """Return true if the features need to be reranked"""
        return (not self.rerank_at_packet and not self.rerank_at_feature) or\
            (self.rerank_at_packet and self.rerank_at_feature and self.packets_per_session == self.rerank_at_packet and
                self.features == self.rerank_at_feature) or\
            (not self.rerank_at_packet and self.rerank_at_feature and self.features == self.rerank_at_feature) or\
            (not self.rerank_at_feature and self.rerank_at_packet and self.packets_per_session ==
             self.rerank_at_packet)

    @property
    def is_load_previous_rank(self):
        """Return true if needed to load the previous features"""
        return self.rerank_at_feature == self.features and self.rerank_at_packet != self.packets_per_session


@dataclass
class ProcessingData(BaseProcessingData):
    indexes_benign: List = field(default_factory=list)
    indexes_malicious: List = field(default_factory=list)


@dataclass
class LucidAnalysisState(AnalysisState):
    current_features: Type[LucidFeaturesHolder] = field(default=None)
    params: LucidDeParams = field(default=None)


class LucidTrafficAnalyser(TrafficAnalyser):
    """Traffic Analyser class that implements the extraction mechanism and the
    termination of a time window while processing data."""
    analysis_state_cls: Type[AnalysisState] = LucidAnalysisState

    def _extract(self, sess_id: Type[BaseKey], eth: Ethernet):
        self.current_session_map[sess_id].value.append(
            tuple(y.create(eth) for y in self.analysis_state.current_features.value))

    def _terminate_timewindow_preprocessing(self):
        if not self.current_session_map:
            return
        asd = next(x for x in self.attackers.values())
        for i, k in enumerate(self.current_session_map):
            target = "benign"
            if k.cast(asd[0]) in asd[1]:
                target = "malicious"
            setattr(self.processing_stats, f"tot_{target}_packets",
                    getattr(self.processing_stats, f"tot_{target}_packets") + self.current_session_map[k].metered_packets +
                    self.current_session_map[k].unmetered_packets)

            if k not in self.seen_sessions_previous_prediction:
                setattr(self.processing_stats, f"unique_{target}", getattr(
                    self.processing_stats, f"unique_{target}") + 1)
                self.seen_sessions_previous_prediction[k] = True
            setattr(self.processing_stats, f"tot_{target}", getattr(
                self.processing_stats, f"tot_{target}") + 1)
            getattr(self.processing_stats, f"indexes_{target}").append(i)
        preprocessed, _, _ = self.de.preprocess_samples(self)
        with h5py.File(f"{self.dump_path}.h5", 'a') as hf:
            for ttype in ["benign", "malicious"]:
                if not len(getattr(self.processing_stats, f"indexes_{ttype}")):
                    continue
                vals = preprocessed[getattr(
                    self.processing_stats, f"indexes_{ttype}")]
                getattr(self.processing_stats, f"indexes_{ttype}").clear()
                if ttype in hf:
                    hf[ttype].resize(
                        (hf[ttype].shape[0] + vals.shape[0]), axis=0)
                    hf[ttype][-vals.shape[0]:] = vals
                else:
                    hf.create_dataset(ttype, data=vals,
                                      maxshape=(None, *vals.shape[1:]))


class LucidCnn(DetectionEngine):
    """Detection Engine composed of:
    - Lucid traffic processing mechanism
    - CNN as detection model
    - Custom ranking mechanism
    """

    features_holder_cls = LucidFeaturesHolder
    traffic_analyser_cls = LucidTrafficAnalyser
    processing_data_cls = ProcessingData
    de_params = LucidDeParams

    def __init__(self, analysis_state: LucidAnalysisState, base_dir) -> None:
        super().__init__(analysis_state, base_dir)
        self.analysis_state: LucidAnalysisState
        self.maxs: np.ndarray = None
        self.indexes = None
        self.adjust_timestamp: int = None
        self._init_scale_method()

    def predict(
            self, ta: LucidTrafficAnalyser) -> Tuple[int, np.ndarray]:
        data, conversion_time, preprocessing_time = self.preprocess_samples(ta)
        predict_time = time.clock_gettime_ns(time.CLOCK_PROCESS_CPUTIME_ID)
        predictions = self.model.predict(
            data, batch_size=self.analysis_state.params.batch_size, verbose=0)
        predict_time = time.clock_gettime_ns(
            time.CLOCK_PROCESS_CPUTIME_ID) - predict_time
        predictions = np.squeeze(predictions, axis=1).astype(np.float64)
        return round(data.nbytes/len(predictions)), predictions, conversion_time, preprocessing_time, predict_time

    @classmethod
    def model_name(cls, packets_per_session: int, features: int = None, **kwargs):
        if not features:
            features = len(cls.features_holder_cls.ALLOWED)
        return "{}p-{}f".format(packets_per_session, features)

    @staticmethod
    @lru_cache
    def _internal_load(basename):
        """Internal method to load the dataset asynchronously"""
        ret = []
        for name in ("train", "validation", "test"):
            with h5py.File(os.path.join(basename, f"{name}.h5"), 'r') as dataset:
                x = dataset["set_x"][:].astype(np.float64)
                y = dataset["set_y"][:].astype(np.float64)
            ret.append(list(sort_keep_comb(x, y)))
        return ret

    @staticmethod
    def _load_dataset(basename, features_holder: LucidFeaturesHolder, packets_per_session=None,
                      max_packets_per_session=None, max_features=None, **kwargs):
        """Load dataset and format it according the current parameters features/packets"""
        not_current_indexes = [i for i, k in enumerate(features_holder.ALLOWED)
                               if k not in features_holder.value]
        current_indexes = [i for i, k in enumerate(features_holder.ALLOWED)
                           if k in features_holder.value]

        ret = copy.deepcopy(LucidCnn._internal_load(basename))
        # if max_packets or max_features then keep their position but with the
        # value in that coord set to 0
        for i in range(len(ret)):
            if not packets_per_session:  # Ho solo features aggregate, non pacchetti
                if max_features:
                    ret[i][0][:, not_current_indexes] = 0.0
                else:
                    ret[i][0] = ret[i][0][:, current_indexes]
            else:  # Ho anche pacchetti come dimensione
                if max_packets_per_session:
                    ret[i][0] = ret[i][0][:, :max_packets_per_session, :]
                    ret[i][0][:, packets_per_session:, :] = 0.0
                else:
                    ret[i][0] = ret[i][0][:, :packets_per_session, :]
                if max_features:
                    ret[i][0][:, :, not_current_indexes] = 0.0
                else:
                    ret[i][0] = ret[i][0][:, :, current_indexes]
        return ret[0], ret[1], ret[2]

    def load_model(self):
        current_name = self.model_name(**self.analysis_state.params.__dict__)
        self.analysis_state.current_features = self.features_holder_cls(**load_json_data(
            os.path.join(self.base_dir, current_name, "relevance.json")))
        self.analysis_state.params = LucidDeParams(**load_json_data(os.path.join(
            self.base_dir, current_name, "params.json")))

        max_par = {
            "packets_per_session": self.analysis_state.params.max_packets_per_session or
            self.analysis_state.params.packets_per_session,
            "features": self.analysis_state.params.max_features or
            self.analysis_state.params.features}
        self.model = self._get_arch(
            **max_par, **{k: v for k, v in self.analysis_state.params.__dict__.items() if k not in max_par})
        self.model.load_weights(os.path.join(
            self.base_dir, self.model_name(**max_par), "weights.h5"))
        self._init_scale_method()

    def preprocess_samples(self, ta: LucidTrafficAnalyser):
        is_nested = isinstance(
            next(x for x in ta.current_session_map.values()).value, list)

        # convert data into input-compliant
        conversion_time = time.clock_gettime_ns(time.CLOCK_PROCESS_CPUTIME_ID)
        if is_nested:
            data = np.zeros((
                len(ta.current_session_map),
                self.analysis_state.params.max_packets_per_session or self.analysis_state.params.packets_per_session,
                self.analysis_state.params.max_features or self.analysis_state.params.features),
                dtype=np.float64)
            for i, v in enumerate(ta.current_session_map.values()):
                for ii, vv in enumerate(v.value):
                    for iii, vvv in zip(self.indexes, vv):
                        data[i, ii, iii] = vvv.value
        else:
            data = np.zeros((len(ta.current_session_map),
                             self.analysis_state.params.max_features or self.analysis_state.params.features),
                            dtype=np.float64)
            for i, v in enumerate(ta.current_session_map.values()):
                for ii, vv in zip(self.indexes, v.value):
                    data[i, ii] = vv.value

        conversion_time = time.clock_gettime_ns(
            time.CLOCK_PROCESS_CPUTIME_ID) - conversion_time
        preprocessing_time = time.clock_gettime_ns(
            time.CLOCK_PROCESS_CPUTIME_ID)
        # adjust timestamp
        if self.adjust_timestamp:
            data[:, :, self.adjust_timestamp] -= data[:,
                                                      [0], self.adjust_timestamp]
        # set min to zero
        data[data < 0] = 0.0
        # scale between 0 and max value
        data[..., :] /= self.maxs
        # remove nan
        np.nan_to_num(data, copy=False)
        preprocessing_time = time.clock_gettime_ns(
            time.CLOCK_PROCESS_CPUTIME_ID) - preprocessing_time
        return data, conversion_time, preprocessing_time

    @classmethod
    def _compute_features_weights(
            cls, features_holder: LucidFeaturesHolder, x: np.ndarray, y: np.ndarray,
            model, malicious_threshold, batch_size, metric, only_active=True):
        """Method to rank the features. Given this metric, the method
        computes the rank of each feature by:
        1. Computing the baseline value of the metric with the entire input set as it is
        2. For each feature, the algorithm sets that feature to zero and computes the new metric
        3. The rank of each feature is given by the performance loss/gain given by the difference
        between the new value and the baseline."""
        baseline = TrainMetric(
            _threshold=malicious_threshold, _ytrue=y,
            _ypred=np.squeeze(model.predict(
                x, batch_size=batch_size, verbose=0), axis=1).astype(np.float64))
        minimize = metric in ("log_loss", "fp", "fpr", "fn", "fnr")

        if only_active:
            indexes = list(range(features_holder.n_current))
        else:
            indexes = [i for i, k in enumerate(
                features_holder.ALLOWED) if k in features_holder.value]
        for i, k in zip(indexes, features_holder.value):
            x_tmp = np.copy(x)
            x_tmp[..., i] = 0.0

            t = TrainMetric(
                _threshold=malicious_threshold, _ytrue=y,
                _ypred=np.squeeze(model.predict(
                    x_tmp, batch_size=batch_size, verbose=0), axis=1).astype(np.float64))
            v_base = getattr(baseline, metric)
            v_current = getattr(t, metric)
            if v_base == 0 or v_current == 0:
                # fallback accuracy metric as not possible to compute
                _logger.info(f"Metric for feature {k} switched to fallback accuracy,"
                             f" as baseline={v_base} and current={v_current}")
                features_holder.value[k] = baseline.accuracy - t.accuracy
            elif minimize:
                features_holder.value[k] = v_current - v_base
            else:
                features_holder.value[k] = v_base - v_current

    @classmethod
    def parameters(cls, model=None, features=None, packets_per_session=None,
                   max_features=None, max_packets_per_session=None, **params):
        """Method to return the number of trainable parameters of the model."""
        if model is None:
            model = cls._get_arch(packets_per_session=max_packets_per_session or packets_per_session,
                                  features=max_features or features, **params)
        return int(np.sum([tf.keras.backend.count_params(p) for p in model.trainable_weights]))

    @classmethod
    def _get_arch(
            cls, packets_per_session: int = None, features: int = None,
            max_packets_per_session: int = None, max_features: int = None,
            kernels: int = 64, dropout: float = 0.2,
            regularization: str = "l2", learning_rate: float = 0.001, **kwargs):
        """Method that defines the architecture of the CNN model"""
        if max_packets_per_session:
            packets_per_session = max_packets_per_session
        if max_features:
            features = max_features

        model = tf.keras.models.Sequential([
            tf.keras.layers.Reshape((packets_per_session, features, 1),
                                    input_shape=(packets_per_session, features)),
            # kernel size = (minimo tra n° packets e 3, n° features) - input shape = (n°pacchetti, n°features, 1)
            tf.keras.layers.Conv2D(kernels, (min(packets_per_session, 3), features),
                                   kernel_regularizer=regularization, name='Conv2D'),
            tf.keras.layers.Dropout(dropout, name="Dropout"),
            tf.keras.layers.Activation(tf.keras.activations.relu, name="ReLu"),
            # pool size = (massimo tra 1 e n°pacchetti-2, 1)
            tf.keras.layers.GlobalMaxPooling2D(name="MaxPooling"),
            tf.keras.layers.Flatten(name="Flatten"),
            tf.keras.layers.Dense(1, name='FinalDense'),
            tf.keras.layers.Activation(
                tf.keras.activations.sigmoid, name="Sigmoid")
        ], name=cls.model_name(packets_per_session=packets_per_session, features=features))

        if learning_rate:
            model.compile(loss=tf.keras.metrics.binary_crossentropy,
                          optimizer=tf.keras.optimizers.Adam(
                              learning_rate=learning_rate),
                          metrics=["accuracy"])
        return model

    def _init_scale_method(self):
        """Method to compute the max value of each feature for the scaling, and set the max value of the
        Time feature to the time window"""
        Time.limit = self.analysis_state.time_window

        # check if need to keep features indexes also for those inactive
        if self.analysis_state.params.max_features is not None:
            self.indexes = [i for i, f in enumerate(
                self.features_holder_cls.ALLOWED) if f in self.analysis_state.current_features.value]
            self.maxs = np.array(
                [x.limit for x in self.analysis_state.current_features.ALLOWED], dtype=np.float64)
            self.adjust_timestamp = next((i for i, k in enumerate(
                self.analysis_state.current_features.ALLOWED) if k == Time), None)
        else:
            self.indexes = list(
                range(self.analysis_state.params.features))
            self.maxs = np.array(
                [x.limit for x in self.analysis_state.current_features.value], dtype=np.float64)
            self.adjust_timestamp = next((i for i, k in enumerate(
                self.analysis_state.current_features.value) if k == Time), None)

    @classmethod
    def append_to_dataset(cls, source, dest, ttype, label, indexes=None):
        source += ".h5"
        dest += ".h5"
        with h5py.File(source, 'r') as dataset, h5py.File(dest, 'a') as new_dataset:
            t = dataset[ttype] if ttype in dataset else dataset["set_x"]
            plus_shape = t.shape[0] if not indexes else len(indexes)
            if 'set_x' in new_dataset:
                new_dataset['set_x'].resize(
                    (new_dataset['set_x'].shape[0] + plus_shape), axis=0)
                new_dataset['set_x'][-plus_shape:] = t if not indexes else t[indexes]
                new_dataset['set_y'].resize(
                    (new_dataset['set_y'].shape[0] + plus_shape), axis=0)
                new_dataset['set_y'][-plus_shape:
                                     ] = np.array([label]*plus_shape, dtype=np.float64)
            else:
                new_dataset.create_dataset('set_x', data=t if not indexes
                                           else t[indexes], maxshape=(None, *t.shape[1:]))
                new_dataset.create_dataset('set_y', data=np.array(
                    [label]*plus_shape, dtype=np.float64), maxshape=(None,))
        return None

    @classmethod
    def train(cls, dataset_path: str, models_dir, train_param: LucidDeParams,
              packets_per_session: int, features: int):
        previous = train_param.previous_one(packets_per_session, features)
        # start from a previous features holder and pop less relevant feature
        # untill the current number of features is matched
        if previous is False:
            features_holder = cls.features_holder_cls()
        else:
            pname = cls.model_name(packets_per_session=previous[0],
                                   features=previous[1])
            features_holder = cls.features_holder_cls(**load_json_data(
                os.path.join(models_dir, os.pardir, pname, "relevance.json")))
            to_pop = previous[2]
            while to_pop:
                features_holder.pop_less_relevant(
                    **train_param.__dict__)
                to_pop -= 1

        hyper = {k.name: getattr(train_param, k.name) for k in fields(
            train_param) if isinstance(k.default, (tuple, list))}

        # set current pair of trained values
        train_param = replace(
            train_param, **{"packets_per_session": packets_per_session, "features": features})

        set_seed()
        _logger.info("Loading dataset")
        # load the dataset according to the current parameters
        (xt, yt, _), (xv, yv, _), (xts, yts, pt) = cls._load_dataset(
            dataset_path, features_holder, packets_per_session=packets_per_session,
            max_packets_per_session=train_param.max_packets_per_session,
            max_features=train_param.max_features)
        set_seed()

        name = cls.model_name(
            packets_per_session=packets_per_session, features=features)

        combs = get_all_dict_comb(hyper)
        n_len = len(combs)

        if train_param.is_to_train:
            if n_len > 1:
                n_comb = get_best_comb(n_len)
                n_dispatch = get_best_dispatch(
                    xt.nbytes+yt.nbytes+xv.nbytes+yv.nbytes)
                _logger.info("RandomizedSearchCV model {} on {} combinations over {} and dispatching {} jobs".
                             format(name, n_comb, n_len, n_dispatch))
                rnd_search_cv = RandomizedSearchCV(
                    tf.keras.wrappers.scikit_learn.KerasClassifier(
                        build_fn=cls._get_arch, packets_per_session=train_param.max_packets_per_session or packets_per_session,
                        features=train_param.max_features or features, verbose=0),
                    hyper, cv=[(slice(None), slice(None))],
                    n_iter=n_comb, refit=False, verbose=0, n_jobs=-1, pre_dispatch=n_dispatch)
                rnd_search_cv.fit(xt, yt, epochs=train_param.epochs, validation_data=(xv, yv), callbacks=[
                    tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode="min", patience=train_param.max_loss)])
                hyper = rnd_search_cv.best_params_
            else:
                hyper = combs[0]

            [setattr(train_param, k, v) for k, v in hyper.items()]

            _logger.info(
                f"Fitting Model {name} using best parameters")
            best_model = cls._get_arch(packets_per_session=train_param.max_packets_per_session or packets_per_session,
                                       features=train_param.max_features or features, **hyper)

            hs = best_model.fit(xt, yt,
                                validation_data=(xv, yv),
                                batch_size=train_param.batch_size,
                                epochs=train_param.epochs, verbose=1,
                                callbacks=[
                                    tf.keras.callbacks.EarlyStopping(
                                        monitor='val_loss', mode="min", patience=train_param.max_loss, min_delta=0.01),
                                    tf.keras.callbacks.ModelCheckpoint(
                                        filepath=os.path.join(
                                            models_dir, "weights.h5"),
                                        monitor='val_loss', mode='min',
                                        save_best_only=True, save_weights_only=True)]).history
        else:
            hyper = {k: v for k, v in load_json_data(os.path.join(models_dir, os.pardir, cls.model_name(
                packets_per_session=train_param.max_packets_per_session or packets_per_session,
                features=train_param.max_features or features), "params.json")).items() if k in hyper}
            [setattr(train_param, k, v) for k, v in hyper.items()]
            best_model = cls._get_arch(packets_per_session=train_param.max_packets_per_session or packets_per_session,
                                       features=train_param.max_features or features, **hyper)
            hs = []
            models_dir = os.path.join(models_dir, os.pardir, cls.model_name(
                packets_per_session=train_param.max_packets_per_session, features=train_param.max_features or features))

        if train_param.is_to_rerank:
            _logger.info(f"Computing features importances for {name}")
            cls._compute_features_weights(features_holder, xv, yv, best_model,
                                          train_param.malicious_threshold, train_param.batch_size,
                                          metric=train_param.rank_metric,
                                          only_active=train_param.max_features is None)

        if train_param.is_load_previous_rank:
            previous = cls.model_name(packets_per_session=train_param.rerank_at_packet,
                                      features=train_param.rerank_at_feature)
            features_holder = features_holder.__class__(**load_json_data(
                os.path.join(models_dir, os.pardir, previous, "relevance.json")))

        best_model.load_weights(os.path.join(models_dir, "weights.h5"))

        _logger.info(f"Testing {name}")
        y_pred = np.squeeze(best_model.predict(
            xts, batch_size=train_param.batch_size, verbose=0), axis=1).astype(np.float64)

        return hs, train_param, TrainMetric(
            _threshold=train_param.malicious_threshold,
            _ypred=y_pred, _ytrue=yts), features_holder, pt
