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
Main file for Splitting the provided traffic captures in a smaller size.
Each directory is considered a category containing certain type of traffic
(e.g., benign-traffic, malicious-ddos, malicious-sqli).
Each pcap is split in smaller pcaps of the provided size: is its dimension is not
at least 3 times the provided size, then the pcap is split in 3 using its own dimension.
This is to prevent having unfair split of the pcap within the Train, Val, and Test set
of the dataset.
"""
import argparse
import math
import multiprocessing
import os
from dataclasses import dataclass

from pypacker.ppcap import Reader, Writer

from .lib.utility import create_dir, dump_json_data, get_logger

_logger = get_logger(__name__)


@dataclass
class CaptureConfig:
    path: str
    size: int = 0


# could have used tcpdump -r {} -w {} -C {}, but don't want external dependencies
def _splitter(src_pcap, dst_pcap, pcap_size):
    i = curr_bytes = dump_bytes = 0
    buf = []
    w = Writer(f"{dst_pcap}{i}.pcap")
    _logger.info("Splitting {} in {}".format(src_pcap, pcap_size))
    for ts, raw in Reader(src_pcap):
        buf.append((ts, raw))
        curr_bytes += len(raw) + 16  # 16 bytes of timestamp
        dump_bytes += len(raw) + 16
        if dump_bytes > 2 * 1024**2 or curr_bytes >= pcap_size:  # dump data every 1MB of buffer
            for x, y in buf:
                w.write(y, ts=x)
            buf.clear()
            dump_bytes = 0
        if curr_bytes >= pcap_size:
            w.close()
            curr_bytes = 0
            i += 1
            w = Writer(f"{dst_pcap}{i}.pcap")
    if buf:
        for x, y in buf:
            w.write(y, ts=x)
    w.close()
    _logger.info("Finished {}".format(src_pcap))


def main(args_list):
    # registering cli parameters, to be shown with the -h flag
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        'path', help='capture directory', type=str)
    parser.add_argument(
        '-s', '--size', help='size to truncate in bytes', type=int, default=300*1024**2)
    args = parser.parse_args(args_list).__dict__

    conf = CaptureConfig(path=args["path"], size=args["size"])

    dump_json_data(conf, os.path.join(conf.path, "conf.json"))

    pcaps = [x.replace(".pcap", "")
             for x in os.listdir(conf.path) if x.endswith(".pcap")]

    star_tasks = []
    for cat in pcaps:
        dst_dir = os.path.join(conf.path, cat)
        create_dir(dst_dir, overwrite=True)
        src_pcap = os.path.join(conf.path, "{}.pcap".format(cat))
        dst_pcap = os.path.join(dst_dir, cat)
        pcap_size = os.path.getsize(src_pcap)
        # if pcap size is not at least 3 times the provided size, then
        # split the pcap in 3 according to the pcap size/3
        # otherwise, split pcaps using the provided size
        if pcap_size / conf.size < 3:
            pcap_size = math.ceil(pcap_size/3)
        else:
            pcap_size = conf.size
        star_tasks.append((src_pcap, dst_pcap, pcap_size))

    with multiprocessing.Pool(maxtasksperchild=1) as pool:
        pool.starmap(_splitter, star_tasks)
