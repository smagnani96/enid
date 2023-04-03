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
File containing the definitions of all the field and keys usable withing this framework.
"""
import ipaddress
import socket
import sys
from abc import ABC, abstractclassmethod, abstractmethod
from dataclasses import dataclass, fields, field
from typing import ClassVar, Tuple, Type

from pypacker.layer3.ip import IP
from pypacker.layer4.tcp import TCP
from pypacker.layer4.udp import UDP
from pypacker.layer12.ethernet import Ethernet

from .metrics import ComputationalRequirements


class SourceIP(int):
    """Type representing the Source IP address"""
    def __new__(cls, x=0, *args, **kwargs):
        if isinstance(x, (str, bytes)):
            x = int(ipaddress.IPv4Address(x))
        elif isinstance(x, Ethernet):
            x = cls.extract(x)
        return int.__new__(cls, x, *args, **kwargs)

    @staticmethod
    def extract(eth: Ethernet):
        return SourceIP(eth[IP].src)

    def __repr__(self):
        return str(ipaddress.IPv4Address(self))


class DestinationIP(SourceIP):
    """Type representing the Destination IP address"""
    @staticmethod
    def extract(eth: Ethernet):
        return DestinationIP(eth[IP].dst)


class IPProtocol(int):
    """Type representing the L4 protocol"""
    PROTOCOLS: ClassVar[dict] = {int(num): name[len("IPPROTO_"):] for name, num in vars(
        socket).items() if name.startswith("IPPROTO_")}

    def __new__(cls, x=0, *args, **kwargs):
        if isinstance(x, str):
            x = next(k for k, v in IPProtocol.PROTOCOLS.items()
                     if v == x.upper())
        elif isinstance(x, Ethernet):
            x = cls.extract(x)
        return int.__new__(cls, x, *args, **kwargs)

    @staticmethod
    def extract(eth: Ethernet):
        return IPProtocol(eth[IP].p)

    def __repr__(self):
        return IPProtocol.PROTOCOLS[self]


class SourcePort(int):
    """Type representing the Source L4 port"""
    def __new__(cls, x=0, *args, **kwargs):
        if isinstance(x, str):
            x = int(x)
        elif isinstance(x, Ethernet):
            x = cls.extract(x)
        return int.__new__(cls, x, *args, **kwargs)

    @staticmethod
    def extract(eth: Ethernet):
        if eth[TCP]:
            return SourcePort(eth[TCP].sport)
        elif eth[UDP]:
            return SourcePort(eth[UDP].sport)
        return SourcePort(0)


class DestinationPort(SourcePort):
    """Type representing the Destination L4 port"""
    @staticmethod
    def extract(eth: Ethernet):
        if eth[TCP]:
            return DestinationPort(eth[TCP].dport)
        elif eth[UDP]:
            return DestinationPort(eth[UDP].dport)
        return DestinationPort(0)


class Dataset(str):
    """Type representing the Dataset. Note that this
    field is important, especially when mixing datasets
    in order to avoid conflicts of session identifiers within 2
    different datasets, which may be of different natures."""
    def __new__(cls, x="", *args, **kwargs):
        if isinstance(x, Ethernet):
            x = cls.extract(x)
        return str.__new__(cls, x, *args, **kwargs)

    @staticmethod
    def extract(eth: Ethernet):
        return Dataset(eth.dataset)


class Category(str):
    """Type representing the Category."""
    def __new__(cls, x="", *args, **kwargs):
        if isinstance(x, Ethernet):
            x = cls.extract(x)
        return str.__new__(cls, x, *args, **kwargs)

    @staticmethod
    def extract(eth: Ethernet):
        return Category(eth.category)


class Pcap(str):
    """Type representing the Pcap."""
    def __new__(cls, x="", *args, **kwargs):
        if isinstance(x, Ethernet):
            x = cls.extract(x)
        return str.__new__(cls, x, *args, **kwargs)

    @staticmethod
    def extract(eth: Ethernet):
        return Pcap(eth.pcap)


@dataclass(frozen=True)
class BaseKey(ABC):
    """Base key class, containing at least the dataset, category and pcap fields.
    Note that, unless the dataset, the pcap and category are not used for hashing and
    comparison, as it is supposed that within the same dataset a sessions is always the
    same and does not change in nature (e.g., from benign to malicious)."""
    dataset: Dataset
    category: Category = field(hash=False, compare=False)
    pcap: Pcap = field(hash=False, compare=False)

    @classmethod
    def extract(cls, eth: Ethernet):
        return cls.create(**{k.name: k.type.extract(eth) for k in fields(cls)})

    @property
    @abstractmethod
    def computational_requirements(cls) -> Tuple[ComputationalRequirements]:
        """Method to return the memory requirements for extracting the current ones"""
        raise NotImplementedError()

    @property
    @abstractmethod
    def memory_requirements(self) -> int:
        """Method to return the memory requirements for extracting the current ones"""
        pass

    @abstractclassmethod
    def create(cls, dataset: Dataset, category: Category, pcap: Pcap, **kwargs):
        raise NotImplementedError()

    def cast(self, other_cls):
        if self.__class__ == other_cls:
            return self
        return other_cls.create(**{k.name: getattr(self, k.name, None) for k in fields(other_cls)})

    def to_json(self):
        return {x.name: str(getattr(self, x.name)) for x in fields(self)}


@dataclass(frozen=True)
class NoKey(BaseKey):
    """Identifier that groups all the traffic into a unique stats"""
    computational_requirements: ClassVar[Tuple[ComputationalRequirements]] = tuple(
    )
    memory_requirements: ClassVar[int] = 0

    @classmethod
    def create(cls, dataset, category, pcap, **kwargs):
        return cls(Dataset(dataset), Category(category), Pcap(pcap))


@dataclass(frozen=True)
class SingleIPKey(BaseKey):
    """Identifier that group the traffic by the source IP address"""
    ip: SourceIP
    computational_requirements: ClassVar[Tuple[ComputationalRequirements]] = (ComputationalRequirements.REQUIRED_L3.value,
                                                                              ComputationalRequirements.BASE_MATH_OP.value)
    memory_requirements: ClassVar[int] = 4

    @classmethod
    def create(cls, dataset, category, pcap, ip, **kwargs):
        return cls(Dataset(dataset), Category(category), Pcap(pcap), SourceIP(ip))


@dataclass(frozen=True)
class TwoIPsKey(SingleIPKey):
    """Identifier that group the traffic by the source and destination IP addresses"""
    ip1: DestinationIP

    computational_requirements: ClassVar[Tuple[ComputationalRequirements]] = SingleIPKey.computational_requirements + \
        (ComputationalRequirements.BASE_MATH_OP.value,
         ComputationalRequirements.HASH_COMPUTATION.value)
    memory_requirements: ClassVar[int] = SingleIPKey.memory_requirements + 4

    @classmethod
    def create(cls, dataset, category, pcap, ip, ip1, **kwargs):
        ip, ip1 = SourceIP(ip), DestinationIP(ip1)
        if ip1 > ip:
            ip, ip1 = ip1, ip
        return cls(Dataset(dataset), Category(category), Pcap(pcap), ip, ip1)


@dataclass(frozen=True)
class TwoIPsProtoKey(TwoIPsKey):
    """Identifier that group the traffic by the source and destination IP addresses plus the L4 protocol"""
    proto: IPProtocol

    computational_requirements: ClassVar[Tuple[ComputationalRequirements]] = TwoIPsKey.computational_requirements + \
        (ComputationalRequirements.BASE_MATH_OP.value, )
    memory_requirements: ClassVar[int] = TwoIPsKey.memory_requirements + 1

    @classmethod
    def create(cls, dataset, category, pcap, ip, ip1, proto, **kwargs):
        ip, ip1 = SourceIP(ip), DestinationIP(ip1)
        if ip1 > ip:
            ip, ip1 = ip1, ip
        return cls(Dataset(dataset), Category(category), Pcap(pcap), ip, ip1, IPProtocol(proto))


@dataclass(frozen=True)
class TwoIPsProtoPortsKey(TwoIPsProtoKey):
    """Identifier that group the traffic by the source and destination IP addresses,
    source and destination L4 ports and the L4 protocol"""
    port: SourcePort
    port1: DestinationPort

    computational_requirements: ClassVar[Tuple[ComputationalRequirements]] = TwoIPsProtoKey.computational_requirements +\
        (ComputationalRequirements.BASE_MATH_OP.value,)
    memory_requirements: ClassVar[int] = TwoIPsProtoKey.memory_requirements + 4

    @classmethod
    def create(cls, dataset, category, pcap, ip, ip1, port, port1, proto, **kwargs):
        ip, ip1, port, port1 = SourceIP(ip), DestinationIP(
            ip1), SourcePort(port), DestinationPort(port1)
        if ip1 > ip:
            ip, ip1, port, port1 = ip1, ip, port1, port
        return cls(Dataset(dataset), Category(category), Pcap(pcap), ip, ip1, IPProtocol(proto), port, port1)


def str_to_key(name) -> Type[BaseKey]:
    """Function to return the Key class corresponding to the name provided"""
    ret = getattr(sys.modules[__name__], name)
    if not issubclass(ret, BaseKey):
        raise ValueError(f"{name} is not a BaseKey instance")
    return ret
