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
"""File defining the LucidMlp Detection Engine"""
from dataclasses import dataclass, field

import numpy as np

from .lucid_cnn import LucidCnn, LucidDeParams, tf


@dataclass
class MlpDeParams(LucidDeParams):
    """Class defining parameters for the Detection Model"""
    regularization: bool = field(default=None, init=False, repr=False)


class LucidMlp(LucidCnn):
    """Detection Engine composed of:
    - Lucid traffic processing mechanism
    - Mlp (Fully-Connected) as model architecture
    - Same training parameters and combination defined in LucidCnn
    """
    de_params = MlpDeParams

    @staticmethod
    def _fed_adapt_layer(src_w, dst_w, global_f, local_f, pad=False):
        if src_w.shape == dst_w.shape:
            return src_w
        indexes = [i for i, f in enumerate(global_f) if f in local_f]
        if pad:
            ret = np.zeros(
                dst_w.shape if dst_w.shape[0] > src_w.shape[0] else src_w)
            ret[indexes, ...] = src_w[indexes, ...]
            return ret
        return src_w[indexes, ...]

    @classmethod
    def _get_arch(
            cls, packets_per_session: int = None, features: int = None,
            max_packets_per_session: int = None, max_features: int = None,
            kernels: int = 64, dropout: float = 0.2,
            learning_rate: float = 0.001, **kwargs):
        if max_packets_per_session:
            packets_per_session = max_packets_per_session
        if max_features:
            features = max_features

        model = tf.keras.models.Sequential([
            tf.keras.layers.InputLayer(
                input_shape=(packets_per_session, features), name="Input"),
            tf.keras.layers.Flatten(name="Flatten"),
            tf.keras.layers.Dense(kernels, name='TARGET'),
            tf.keras.layers.Dropout(dropout, name="Dropout"),
            tf.keras.layers.Activation(
                tf.keras.activations.relu, name="ReLu"),
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
