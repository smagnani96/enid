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
File containing the definitions of the labels for the supported datasets
and the function to load their respective malicious flows' IDs.
"""
import itertools
import multiprocessing
import os

from lxml import etree

from .identifiers import TwoIPsKey, TwoIPsProtoPortsKey


def __internal_2012(xml_file):
    attackers = set()
    for child in etree.parse(xml_file).getroot():
        if child.find('Tag').text == "Normal":
            continue
        protocol_string = child.find('protocolName').text.upper()
        if "TCP" in protocol_string:
            proto = "TCP"
        elif "UDP" in protocol_string:
            proto = "UDP"
        elif "ICMP" in protocol_string:
            proto = "ICMP"
        else:
            continue
        attackers.add(TwoIPsProtoPortsKey.create(
            "IDS2012", "", "",
            ip=child.find('source').text,
            ip1=child.find('destination').text,
            proto=proto,
            port=int(child.find('sourcePort').text),
            port1=int(child.find('destinationPort').text)))
    return attackers


def parse_xml_label_file_IDS2012(dir_path):
    ret = set()
    with multiprocessing.Pool(maxtasksperchild=1) as pool:
        for k in pool.map(__internal_2012, [os.path.join(dir_path, x) for x in os.listdir(dir_path) if ".xml" in x]):
            ret.update(k)
    return list(ret)


ATTACK_LABELS = {
    'IDS2012': parse_xml_label_file_IDS2012,
    'IDS2017': lambda _: tuple(
        TwoIPsKey.create("IDS2017", "", "", ip=x, ip1=y) for x, y in itertools.product(
            ['172.16.0.1'], ['192.168.10.50'])),
    'IDS2018': lambda _: tuple(
        TwoIPsKey.create("IDS2018", "", "", ip=x, ip1=y) for x, y in itertools.product(
            ['18.218.115.60', '18.219.9.1', '18.219.32.43', '18.218.55.126', '52.14.136.135',
                '18.219.5.43', '18.216.200.189', '18.218.229.235', '18.218.11.51', '18.216.24.42'],
            ['18.218.83.150', '172.31.69.28'])),
    'IDS2019': lambda _: tuple(
        TwoIPsKey.create("IDS2019", "", "", ip=x, ip1=y) for x, y in itertools.product(
            ['172.16.0.5'], ['192.168.50.1', '192.168.50.4']))
}
