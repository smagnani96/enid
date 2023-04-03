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
File that defines the features required by Lucid, their extraction mechanisms
and parameters. Also, the FeatureHolder class with the method for popping the
less relevant feature is defined.
"""
import sys
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import ClassVar
from typing import OrderedDict as OrderedDictType
from typing import Tuple, Type

from pypacker.layer3.icmp import ICMP
from pypacker.layer3.ip import IP
from pypacker.layer4.ssl import SSL
from pypacker.layer4.tcp import TCP
from pypacker.layer4.udp import UDP
from pypacker.layer12.arp import ARP
from pypacker.layer12.ethernet import Ethernet
from pypacker.layer567.dns import DNS
from pypacker.layer567.http import HTTP
from pypacker.layer567.telnet import Telnet
from pypacker.pypacker import Packet

from ...definitions import (BaseFeature, BaseKey, ComputationalRequirements,
                            FeaturesHolder)

# Lucid supported protocols used in previous work
_SUPPORTED_PROTOCOLS: Tuple[Type[Packet]] = (
    HTTP, Telnet, SSL, UDP, TCP, IP, ICMP, DNS, Ethernet, ARP)


@dataclass
class HighestLayer(BaseFeature):
    computational_requirements: ClassVar[Tuple[ComputationalRequirements]] = (
        ComputationalRequirements.REQUIRED_L4, ComputationalRequirements.ENHANCED_MATH_OP,
        ComputationalRequirements.HASH_COMPUTATION, ComputationalRequirements.ENHANCED_MATH_OP)
    memory_requirements: ClassVar[int] = 2
    limit: ClassVar[int] = 1 << 10

    def extract(self, eth: Ethernet):
        self.value = 1 << 10 - \
            next(i for i, x in enumerate(_SUPPORTED_PROTOCOLS) if x in eth)


@dataclass
class Protocols(BaseFeature):
    computational_requirements: ClassVar[Tuple[ComputationalRequirements]] = (
        ComputationalRequirements.REQUIRED_L4, ComputationalRequirements.ENHANCED_MATH_OP,
        ComputationalRequirements.HASH_COMPUTATION) +\
        tuple(ComputationalRequirements.BASE_MATH_OP for _ in range(
            len(_SUPPORTED_PROTOCOLS)))
    memory_requirements: ClassVar[int] = 2
    limit: ClassVar[int] = 1 << 10

    def extract(self, eth: Ethernet):
        protocols = tuple(x.__class__ for x in eth)
        self.value = int(
            ''.join("1" if p in protocols else "0" for p in _SUPPORTED_PROTOCOLS), 2)


@dataclass
class Time(BaseFeature):
    computational_requirements: ClassVar[Tuple[ComputationalRequirements]] = (
        ComputationalRequirements.TIMER,)
    memory_requirements: ClassVar[int] = 8
    limit: ClassVar[int] = None

    def extract(self, eth: Ethernet):
        self.value = eth.timestamp


@dataclass
class IcmpType(BaseFeature):
    computational_requirements: ClassVar[Tuple[ComputationalRequirements]] = (
        ComputationalRequirements.REQUIRED_L3, ComputationalRequirements.BASE_MATH_OP)
    memory_requirements: ClassVar[int] = 2
    limit: ClassVar[int] = 1 << 16

    def extract(self, eth: Ethernet):
        if eth[ICMP]:
            self.value = eth[ICMP].type


@dataclass
class UDPLength(BaseFeature):
    computational_requirements: ClassVar[Tuple[ComputationalRequirements]] = (
        ComputationalRequirements.REQUIRED_L4, ComputationalRequirements.BASE_MATH_OP)
    memory_requirements: ClassVar[int] = 2
    limit: ClassVar[int] = 1 << 16

    def extract(self, eth: Ethernet):
        if eth[UDP]:
            self.value = eth[UDP].ulen - 8


@dataclass
class TCPWindow(BaseFeature):
    computational_requirements: ClassVar[Tuple[ComputationalRequirements]] = (
        ComputationalRequirements.REQUIRED_L4, ComputationalRequirements.BASE_MATH_OP)
    memory_requirements: ClassVar[int] = 2
    limit: ClassVar[int] = 1 << 16

    def extract(self, eth: Ethernet):
        if eth[TCP]:
            self.value = eth[TCP].win


@dataclass
class TCPFlags(BaseFeature):
    computational_requirements: ClassVar[Tuple[ComputationalRequirements]] = (
        ComputationalRequirements.REQUIRED_L4, ComputationalRequirements.BASE_MATH_OP)
    memory_requirements: ClassVar[int] = 2
    limit: ClassVar[int] = 1 << 16

    def extract(self, eth: Ethernet):
        if eth[TCP]:
            self.value = eth[TCP].flags


@dataclass
class TCPLength(BaseFeature):
    computational_requirements: ClassVar[Tuple[ComputationalRequirements]] = (
        ComputationalRequirements.REQUIRED_L4, ComputationalRequirements.BASE_MATH_OP)
    memory_requirements: ClassVar[int] = 2
    limit: ClassVar[int] = 1 << 16

    def extract(self, eth: Ethernet):
        if eth[TCP]:
            self.value = eth[IP].len - (eth[IP].hl << 2)


@dataclass
class IPFlags(BaseFeature):
    computational_requirements: ClassVar[Tuple[ComputationalRequirements]] = (
        ComputationalRequirements.REQUIRED_L3, ComputationalRequirements.BASE_MATH_OP)
    memory_requirements: ClassVar[int] = 2
    limit: ClassVar[int] = 1 << 16

    def extract(self, eth: Ethernet):
        self.value = eth[IP].flags


@dataclass
class IPLength(BaseFeature):
    computational_requirements: ClassVar[Tuple[ComputationalRequirements]] = (
        ComputationalRequirements.REQUIRED_L3, ComputationalRequirements.BASE_MATH_OP)
    memory_requirements: ClassVar[int] = 2
    limit: ClassVar[int] = 1 << 16

    def extract(self, eth: Ethernet):
        self.value = eth[IP].len


@dataclass
class LucidFeaturesHolder(FeaturesHolder):
    ALLOWED: ClassVar[Tuple[Type[BaseFeature]]] = (HighestLayer, Protocols, Time,
                                                   IcmpType, UDPLength,
                                                   TCPWindow, TCPFlags, TCPLength,
                                                   IPFlags, IPLength)
    value: OrderedDictType[Type[BaseFeature], float] = field(
        default_factory=OrderedDict)

    def __post_init__(self):
        if not self.value:
            self.value = OrderedDict.fromkeys(self.ALLOWED, None)
        elif isinstance(self.value, (list, tuple)):
            if isinstance(self.value[0], dict):
                self.value = OrderedDict(
                    [(getattr(sys.modules[self.__module__], k["key"]), k["value"]) for k in self.value])
            else:
                self.value = OrderedDict.fromkeys(self.value, None)

    def pop_less_relevant(self, prune_zero_first=False, key_depth_class: Type[BaseKey] = None, **kwargs):
        """Method for popping the less relevant feature from the current ones.
        The method looks iteratively at the following attributes, unless only 1
        feature is remained and removed:
        1. If prune zero first, then consider all those with relevance=0, else
        take all current into account
        2. Get all features with minimum importance
        3. Get all features with max CPU instructions
        4. Get all features with max memory

        If still more than 1 feature remains, then pop the first one.
        """
        def _internal_loop(rest, cond, is_backed):
            tmpk = []
            tmpv = sys.maxsize if cond == 0 else sys.maxsize*-1
            for k in rest:
                if cond == 0:
                    if is_backed:
                        v = self.value[k][1]
                    elif isinstance(self.value[k], (list, tuple)):
                        v = self.value[k][0]
                    else:
                        v = self.value[k]
                elif cond == 1:
                    if key_depth_class is None:
                        v = ComputationalRequirements.requirements_to_cost(
                            k.computational_requirements, ignore_depth=True)
                    else:
                        v = ComputationalRequirements.requirements_to_cost(
                            k.computational_requirements, ignore_depth=key_depth_class)
                elif cond == 2:
                    v = k.memory_requirements
                else:
                    raise Exception()
                if (cond == 0 and v < tmpv) or ((cond == 1 or cond == 2) and v > tmpv):
                    tmpv = v
                    tmpk = [k]
                elif v == tmpv:
                    tmpk.append(k)
            return tmpk

        # used only in case all features' relevances were 0, so a backup metric is used.
        is_backed_up = sum(1 for v in self.value.values()
                           if isinstance(v, (tuple, list))) == len(self.value)

        # get all features with no importance (0)
        if prune_zero_first:
            mink = [k for k, v in self.value.items() if (is_backed_up and v[1] == 0) or
                    (not is_backed_up and v == 0)] or list(self.value.keys())
            if len(mink) == 1:
                return self.value.pop(mink[0])
        else:
            mink = list(self.value.keys())
        # get all features with minimum importance
        mink = _internal_loop(mink, 0, is_backed_up)
        if len(mink) == 1:
            return self.value.pop(mink[0])
        # get all features with max CPU
        mink = _internal_loop(mink, 1, is_backed_up)
        if len(mink) == 1:
            return self.value.pop(mink[0])
        # get all features with max memory
        mink = _internal_loop(mink, 2, is_backed_up)
        return self.value.pop(mink[0])
