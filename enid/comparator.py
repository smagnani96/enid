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
Main program to debug the online test results produced with the
DebugLevel.ENHANCED. Given a certain threshold of packets provided,
this program outputs all the benign flows that have been erroneously
classified and their n° of following packets blocked, and the malicious
sessions not detected and their n° packets that reached the application
in the following intervals.
"""
import argparse
import os

from .datasetter import DatasetConfig
from .lib.identifiers import str_to_key
from .lib.utility import load_json_data


def _load_history(path):
    s = load_json_data(path)
    for k in s:
        key_cls = str_to_key(s[k]["analysis_state"]["current_key"])
        s[k]["session_map"] = {key_cls.create(
            **j["key"]): j["value"] for j in s[k]["session_map"]}
        s[k]["black_map"] = {key_cls.create(
            **j["key"]): j["value"] for j in s[k]["black_map"]}
    return s


def main(args_list):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "files", help="test history files to compare", nargs="2+", type=str)
    parser.add_argument("-t", "--threshold", type=int, default=0)

    args = parser.parse_args(args_list).__dict__
    files = {os.path.basename(x): _load_history(x) for x in args["files"]}
    malicious = {}
    [malicious.update(DatasetConfig(**load_json_data(os.path.join(x, os.pardir,
                      os.pardir, os.pardir, "conf.json"))).attackers) for x in args["files"]]

    already_handled = {}
    timewindows = list(range(0, max(len(x) for x in files.values())))

    for i, k in enumerate(timewindows):
        seen_sess = set()
        [seen_sess.update(v[k]["session_map"].keys()) for v in files.values()]
        for s in seen_sess:
            if s not in already_handled and all(s in v[k]["session_map"] for v in files.values()) and\
                (not all(v[k]["session_map"][s]["prediction"] > 0.5 for v in files.values()) or
                 not all(v[k]["session_map"][s]["prediction"] < 0.5 for v in files.values())):
                already_handled[s] = True
                if getattr(s, "dataset", None):
                    is_malicious_key = s.cast(
                        malicious[s.dataset][0].__class__) in malicious[s.dataset]
                else:
                    is_malicious_key = any(
                        s.cast(p[0].__class__) in p for p in malicious.values())
                print("---------------------")
                print("Detected difference in time interval", k)
                print(f"Key (is_ddos={is_malicious_key}):", s)
                print()
                for k, v in files.items():
                    if v[k]["session_map"][s]["prediction"] <= 0.5:
                        continue
                    fut_handled = sum(
                        v[tw]["black_map"][s] if s in v[tw]["black_map"] else 0 for tw in timewindows[i+1:])

                    if fut_handled < args["threshold"]:
                        continue
                    print("File:", k)
                    print("SessionValue:", v[k]["session_map"][s])
                    print("Features:", [(k["key"], k["value"])
                          for k in v[k]["analysis_state"]["current_features"]["value"]])
                    print("Future Packets Mitigated:", fut_handled, "in the following", len(
                        timewindows[i+1:]), "time intervals")
                    print()
                print("---------------------")
                input()
