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
Main file for creating plots of the achieved results.
This program can create all kinds of plots, according to the
parameters of the Detection Model used and the configuration of the NIDS.
In particular, the following results can be plotted:
1. Models complexity
2. Models training history (loss and accuracy)
3. Models features relevances
4. Models train results
5. Models train results detailed for each type of granularity (dataset/category/pcap)
6. Test results of the various configurations
7. Test results of the various configuration for each type of granularity

Train and Test (both offline and online) results can be plotted by condensing parameters
and creating boxplot, in case of multiple dimensions of the Engine (e.g., packets P and features F)
"""
import argparse
import math
import multiprocessing
import os
from dataclasses import dataclass
from multiprocessing.pool import ThreadPool
from typing import Type

import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from .lib.definitions import DetectionEngine, FeaturesHolder
from .lib.metrics import TestMetric, TrainMetric
from .lib.utility import (create_dir, get_logger, handler,
                          load_json_data, snake_to_camel)
from .trainer import ModelConfig, DatasetConfig

_logger = get_logger(__name__)


@dataclass
class PlotArgs:
    """PlotArgs dataclass to contain personalised arguments for adjusting plots"""
    logx: bool = False
    logy: bool = False
    fit: bool = False
    offlegend: bool = False
    hlegend: bool = False
    seplegend: bool = False
    minx: float = None
    maxx: float = None
    miny: float = None
    maxy: float = None
    xgrid: bool = True
    ygrid: bool = True
    figsizex: float = 4
    figsizey: float = 3
    ylabelchars: int = 35
    xlabelchars: int = 35
    style: str = "science"

    def __post_init__(self):
        plt.style.use(self.style)

    def adjust(self, fig: Figure = None, ax: plt.Axes = None, xlabel: str = None, ylabel: str = None,
               path: str = None, suptitle: str = None, title: str = None):
        if ax:
            if self.xgrid:
                ax.grid(True, axis='x')
            if self.ygrid:
                ax.grid(True, axis='y')
            if self.logy:
                ax.set_yscale("log")
            if self.maxy is not None or self.miny is not None:
                ax.set_ylim(top=self.maxy, bottom=self.miny)
            if self.maxx is not None or self.minx is not None:
                ax.set_xlim(right=self.maxx, left=self.minx)
            if self.logx:
                ax.set_xscale("log")
            if xlabel:
                ax.set_xlabel(xlabel)
            if ylabel:
                ax.set_ylabel(ylabel)
            if title:
                ax.set_title(title)

            if self.offlegend:
                leg = ax.get_legend()
                if leg:
                    leg.remove()
            else:
                params = {}
                if self.hlegend:
                    params = {
                        "loc": "upper center",
                        "ncol": math.ceil(math.sqrt(len(ax.lines))) if not self.seplegend else len(ax.lines),
                        "bbox_to_anchor": (0.5, 1.25) if not self.seplegend else None
                    }
                else:
                    params = {"loc": "center left",
                              "ncol": 1, "bbox_to_anchor": (1, 0.5)}
                if self.seplegend:
                    params.pop("loc")
                    params.pop("bbox_to_anchor")
                ax.legend(**params)
                if self.seplegend:
                    label_params = ax.get_legend_handles_labels()
                    figl = plt.figure(figsize=(3, 2))
                    figl.legend(*label_params, loc="center", ncol=params["ncol"], bbox_to_anchor=(0.5, 0.5),
                                fontsize=40, markerscale=4, handlelength=1.5)
                    figl.savefig(f"{path}_legend.pdf",
                                 bbox_inches='tight', pad_inches=0.0)
                    plt.close(figl)
                    ax.get_legend().remove()
        if fig:
            if self.figsizex:
                fig.set_figwidth(self.figsizex)
            if self.figsizey:
                fig.set_figheight(self.figsizey)
            if suptitle:
                fig.suptitle(suptitle)
            if path:
                fig.savefig(f"{path}.pdf")
                plt.close(fig)


def _labelify(params, exclude):
    """Function to transform Detection model parameters into symbols
    E.g.: features -> f; packets_per_session -> p
    """
    return '-'.join(f"{v}" + "\\textit{" + k[0] + "}" for k, v in params.items() if k != exclude)


def _plottify_metric_name(name: str, max_len):
    """Method to adjust metric name for plotting according to the length provided"""
    name = name.replace("percentage", "_(\\%)_")
    name = name.replace("_per_", "_/_")
    name = snake_to_camel(name, join_char=' ')
    for x in ("Tpr", "Tnr", "Fpr", "Fnr"):
        name = name.replace(x, x.upper())
    if len(name) > max_len:
        try:
            post = name.index(" ", max_len, len(name))
        except ValueError:
            post = len(name)
        name = name[:post] + "\n" + name[post:]
    return name


def _plot_features_relevance(c, de: Type[DetectionEngine], path: str, plot_args: PlotArgs):
    name = de.model_name(**c)
    _logger.info(f"Starting plotting relevance of {name}")
    rel: FeaturesHolder = de.features_holder_cls(**load_json_data(os.path.join(
        path, "models", name, "relevance.json")))
    if not rel:
        _logger.info(f"No relevance for {name}")
        return
    xx, names = [], []
    for x in de.features_holder_cls.ALLOWED:
        asd = rel.get_feature_value(x)
        if asd is not None and not isinstance(asd, (list, tuple)):
            xx.append(asd)
            names.append("\\textbf{" + x.__name__ + "}")
        else:
            xx.append(0)
            names.append(x.__name__)
    fig, ax = plt.subplots()
    ax.barh(names, xx)
    plot_args.adjust(fig=fig, ax=ax, xlabel="Activation Score",
                     path=os.path.join(path, "charts", "models", "features_relevance", name))
    _logger.info(f"Finished plotting relevance of {name}")


def _plot_train_histories(c, de: DetectionEngine, path: str, plot_args: PlotArgs):
    name = de.model_name(**c)
    _logger.info(f"Starting plotting histories of {name}")
    hs = load_json_data(os.path.join(path, "models", name, "history.json"))
    if not hs:
        _logger.info(f"No history for {name}")
        return
    fig, ax = plt.subplots(2, 1, sharex=True)
    fig.subplots_adjust(hspace=.0)
    plot_args.offlegend = True
    ax[0].plot(hs['loss'], 'b', label="Train")
    ax[0].plot(hs["val_loss"], 'r', label="Validation")
    plot_args.adjust(fig=fig, ax=ax[0], ylabel='Loss')
    ax[1].plot(hs['accuracy'], 'b', label="Train")
    ax[1].plot(hs["val_accuracy"], 'r', label="Validation")
    plot_args.adjust(fig=fig, ax=ax[1], xlabel="Epocs", ylabel='Accuracy')
    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(
        0.5, 1), ncol=len(labels), bbox_transform=fig.transFigure)
    plot_args.adjust(fig=fig, path=os.path.join(
        path, "charts", "models", "histories", name))
    _logger.info(f"Finished plotting histories of {name}")


def _plot_models_complexity(agg_by_name, models_conf: ModelConfig, path, plot_args: PlotArgs):
    """Function to plot the model complexity in terms of trainable parameters"""
    _logger.info(
        f"Started plotting complexity using {agg_by_name} as x-axis")
    store_path = os.path.join(
        os.path.dirname(os.path.normpath(path)),
        "charts", "models", "complexity", f"trainable_by_{agg_by_name}")
    fig, ax = plt.subplots()

    values = sorted(getattr(models_conf.train_params, agg_by_name))
    values_tick = list(range(len(values)))
    all_combs = models_conf.train_params.all_train_combs(exclude=agg_by_name)
    ax.set_prop_cycle(cycler(linestyle=[":"]*len(all_combs)) +
                      cycler(color=[plt.cm.nipy_spectral(i) for i in np.linspace(0, 1, len(all_combs))]) +
                      cycler(marker=sorted([x for x, v in Line2D.markers.items() if v != "nothing" and x not in ("|", "_")],
                                           reverse=True, key=lambda x: str(x))[:len(all_combs)]))

    def asd(*args):
        name = models_conf.detection_engine.model_name(**{
            agg_by_name: args[1],
            **args[0]})
        return args[0], args[1], models_conf.detection_engine.parameters(
            **load_json_data(os.path.join(path, name, "params.json")))

    with ThreadPool() as pool:
        r = pool.starmap(asd, [(c, v) for c in all_combs for v in values])
    for c in all_combs:
        tmp = []
        for v in values:
            x = next(x for x in r if x[0] == c and x[1] == v)
            tmp.append(x[2])
        ax.plot(values_tick, tmp, label=_labelify(c, agg_by_name))

    ax.set_xticks(values_tick)
    ax.set_xticklabels(values)
    ax.tick_params(axis='x', which='minor', bottom=False, top=False)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plot_args.adjust(fig=fig, ax=ax, xlabel="\\textit{" + agg_by_name[0] + "}",
                     ylabel="Trainable Parameters", path=store_path)
    _logger.info(
        f"Finished plotting complexity using {agg_by_name} as x-axis")


def _plot_results_metrics(agg_by_name, models_conf: ModelConfig,
                          path, metric, plot_args: PlotArgs, add_name="", depth=tuple()):
    """Function used to print result metrics for both offline and online testing"""
    _logger.info(
        f"Started plotting metric {metric} depth={depth} using {agg_by_name} as x-axis")
    store_name = os.path.join(os.path.dirname(os.path.normpath(
        path)), "charts", os.path.basename(path), "results", add_name)
    for x in depth:
        store_name = os.path.join(store_name, x)
    store_name = os.path.join(
        store_name, f"total_by_{agg_by_name}" if agg_by_name else "", metric)

    fig, ax = plt.subplots()

    values = sorted(getattr(models_conf.train_params, agg_by_name))
    m_name = _plottify_metric_name(metric, plot_args.ylabelchars)
    values_tick = values if plot_args.logx else list(range(len(values)))
    all_combs = models_conf.train_params.all_train_combs(exclude=agg_by_name)
    ax.set_prop_cycle(cycler(linestyle=[":"]*len(all_combs)) +
                      cycler(color=[plt.cm.nipy_spectral(i) for i in np.linspace(0, 1, len(all_combs))]) +
                      cycler(marker=(sorted([x for x, v in Line2D.markers.items() if v != "nothing" and x != "|"],
                                            reverse=True, key=lambda x: str(x))*len(all_combs))[:len(all_combs)]))
    for c in all_combs:
        tmp = []
        for v in values:
            name = models_conf.detection_engine.model_name(
                **c, **{agg_by_name: v})
            j = load_json_data(os.path.join(
                path, name, add_name, "results.json"))
            if not depth:
                tmp.append(j[metric])
            elif len(depth) == 1:
                tmp.append(j["datasets"][depth[0]][metric])
            elif len(depth) == 2:
                tmp.append(j["datasets"][depth[0]]
                           ["categories"][depth[1]][metric])
            else:
                tmp.append(j["datasets"][depth[0]]["categories"]
                           [depth[1]]["captures"][depth[2]][metric])
        if plot_args.fit:
            aasd = ax.scatter(values, tmp, label=_labelify(c, agg_by_name))
            ax.plot(values, np.polyval(np.polyfit(values, tmp, 2),
                    values), color=aasd.get_facecolor()[0])
        else:
            ax.plot(values_tick, tmp, label=_labelify(c, agg_by_name))

    ax.set_xticks(values_tick)
    ax.set_xticklabels(values)
    ax.tick_params(axis='x', which='minor', bottom=False, top=False)
    plot_args.adjust(
        fig=fig, ax=ax, xlabel="\\textit{" + agg_by_name[0] + "}", ylabel=m_name, path=store_name)
    _logger.info(
        f"Finished plotting metric {metric} depth={depth} agg by {agg_by_name}")


def _plot_results_metrics_boxplot(agg_by_name, models_conf: ModelConfig,
                                  path, metric, test_name, plot_args: PlotArgs, depth=tuple()):
    """Function for plotting resulting metrics condensed """
    _logger.info(f"Started Plotting boxplot of {metric}")
    fig, ax = plt.subplots()
    vals = []
    m_name = _plottify_metric_name(metric, plot_args.ylabelchars)
    asd = sorted(getattr(models_conf.train_params, agg_by_name))
    asd_tick = asd if plot_args.logx else list(range(len(asd)))
    for v in sorted(getattr(models_conf.train_params, agg_by_name)):
        tmp = []
        for c in [x for x in models_conf.train_params.all_train_combs() if x[agg_by_name] == v]:
            name = models_conf.detection_engine.model_name(**c)
            j = load_json_data(os.path.join(
                path, test_name, name, "results.json"))
            if not depth:
                tmp.append(j[metric])
            elif len(depth) == 1:
                tmp.append(j["datasets"][depth[0]][metric])
            elif len(depth) == 2:
                tmp.append(j["datasets"][depth[0]]
                           ["categories"][depth[1]][metric])
            else:
                tmp.append(j["datasets"][depth[0]]["categories"]
                           [depth[1]]["captures"][depth[2]][metric])
        vals.append(tmp)
    ret = ax.boxplot(vals, positions=asd_tick)
    if plot_args.fit:
        xs = asd
        ys = np.array([q.get_ydata()[0] for q in ret['medians']])
        ys = ys[~np.isnan(ys)]
        polyval = np.polyval(np.polyfit(xs, ys, 2), xs)
        ax.plot(asd_tick, [polyval[q] for q in asd_tick])
    for v, vv, fl in zip(asd, vals, ret['fliers']):
        off = -0.5
        for c, vvv in zip([x for x in models_conf.train_params.all_train_combs() if x[agg_by_name] == v], vv):
            if vvv in fl.get_ydata():
                ax.text(fl.get_xdata()[0]+off, vvv,
                        _labelify(c, agg_by_name), va='center', ha='left')
                off *= -1
            off = -0.5 if off < 0 else 0.25
    ax.minorticks_off()
    ax.set_xticks(asd_tick)
    ax.set_xticklabels([str(k) for k in asd])
    path = os.path.join(path, "charts", test_name, "results")
    for x in depth:
        path = os.path.join(path, x)
    path = os.path.join(path, f"condensed_by_{agg_by_name}", metric)
    plot_args.adjust(
        fig=fig, ax=ax, xlabel="\\textit{" + agg_by_name[0] + "}", ylabel=m_name, path=path)
    _logger.info(f"Finished Plotting boxplot of {metric} agg by {agg_by_name}")


def main(args_list):
    # registering cli parameters
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'de_models_dir', help='path to the directory containing results', type=str)
    parser.add_argument(
        '-mf', '--model-complexity', help='display features relevance', action="store_true")
    parser.add_argument(
        '-fr', '--features-relevance', help='display features relevance', action="store_true")
    parser.add_argument(
        '-th', '--train-histories', help='display train histories', action="store_true")
    parser.add_argument(
        '-tr', '--train-results', help='display train results', action="store_true")
    parser.add_argument(
        '-trd', '--train-results-detailed', help='display chart per-capture', action="store_true")
    parser.add_argument(
        '-ts', '--test-results', help='display test results', action="store_true")
    parser.add_argument(
        '-tsd', '--test-results-detailed', help='display chart per-capture in test', action="store_true")
    parser.add_argument(
        '-tl', '--transfer-learning', help='transfer learning', action="store_true")
    parser.add_argument(
        '-tld', '--transfer-learning-detailed', help='transfer learning detailed', action="store_true")
    parser.add_argument(
        '-b', '--boxplot', help='condense in boxplot', action="store_true")
    args = parser.parse_args(args_list).__dict__

    dataset_conf: DatasetConfig = DatasetConfig(
        **load_json_data(os.path.join(args["de_models_dir"], os.pardir, "conf.json")))
    models_conf: ModelConfig = ModelConfig(
        **load_json_data(os.path.join(args["de_models_dir"], "models", "conf.json")))

    star_tasks = []
    plot_args = PlotArgs()

    if args["model_complexity"]:
        # print complexity of the model in terms of trainable parameters
        tmp = os.path.join(args["de_models_dir"], "models")
        for k in models_conf.train_params.train_combs():
            if len(getattr(models_conf.train_params, k)) <= 1:
                continue
            create_dir(os.path.join(
                args["de_models_dir"], "charts", "models", "complexity"))
            star_tasks.append(
                (_plot_models_complexity, (k, models_conf, tmp, plot_args)))

    if args["train_histories"] and models_conf:
        # print train histories, showing loss and accuracy of the train vs validation set
        create_dir(os.path.join(
            args["de_models_dir"], "charts", "models", "histories"))
        [star_tasks.append((_plot_train_histories, (c, models_conf.detection_engine, args["de_models_dir"], plot_args)))
         for c in models_conf.train_params.all_train_combs()]

    if args["features_relevance"] and models_conf:
        # print relevance of each feature in each configuration of the NIDS
        create_dir(os.path.join(
            args["de_models_dir"], "charts", "models", "features_relevance"))
        [star_tasks.append((_plot_features_relevance, (c, models_conf.detection_engine, args["de_models_dir"], plot_args)))
         for c in models_conf.train_params.all_train_combs()]

    if args["train_results"]:
        # print train results for each parameter (e.g., feature F in the x-axes, packets P in the x-axes)
        tmp = os.path.join(args["de_models_dir"], "models")
        for k in models_conf.train_params.train_combs():
            if len(getattr(models_conf.train_params, k)) <= 1:
                continue
            create_dir(os.path.join(
                args["de_models_dir"], "charts", "models", "results", f"total_by_{k}"))
            [star_tasks.append((_plot_results_metrics, (k, models_conf, tmp, m, plot_args)))
                for m in TrainMetric.get_metrics()]

            if args["boxplot"] and sum(len(getattr(models_conf.train_params, kk))
                                       for kk in models_conf.train_params.train_combs() if kk != k) > 1:
                # print train results condensing plots by the parameters of interest
                # (e.g., each box in the boxplot refers to a specific features F' and all models with that F'
                # but other different parameters, such as all packets P)
                create_dir(os.path.join(
                    args["de_models_dir"], "charts", "models", "results", f"condensed_by_{k}"))
                [star_tasks.append((_plot_results_metrics_boxplot, (k, models_conf, args["de_models_dir"],
                                                                    m, "models", plot_args)))
                 for m in TrainMetric.get_metrics()]

    if args["train_results_detailed"]:
        tmp = os.path.join(args["de_models_dir"], "models")
        for k in models_conf.train_params.train_combs():
            if len(getattr(models_conf.train_params, k)) <= 1:
                continue
            # print train results detailed at the DATASET granularity for each parameter
            # (e.g., feature F in the x-axes, packets P in the x-axes)
            for dataset_name, v in dataset_conf.offline.datasets.items():
                create_dir(os.path.join(
                    args["de_models_dir"], "charts", "models", "results", dataset_name, f"total_by_{k}"))
                [star_tasks.append((_plot_results_metrics, (k, models_conf, tmp, m, plot_args, "", (dataset_name,))))
                    for m in TrainMetric.get_metrics()]

                # print train results detailed at the DATASET granularity condensing plots by the parameters of interest
                # (e.g., each box in the boxplot refers to a specific features F' and all models with that F'
                # but other different parameters, such as all packets P)
                if args["boxplot"] and sum(len(getattr(models_conf.train_params, kk))
                                           for kk in models_conf.train_params.train_combs() if kk != k) > 1:
                    create_dir(os.path.join(
                        args["de_models_dir"], "charts", "models", "results", dataset_name, f"condensed_by_{k}"))
                    [star_tasks.append((_plot_results_metrics_boxplot, (k, models_conf, args["de_models_dir"],
                                                                        m, "models", plot_args, (dataset_name,))))
                     for m in TrainMetric.get_metrics()]

                # print train results detailed at the CATEGORY granularity for each parameter
                # (e.g., feature F in the x-axes, packets P in the x-axes)
                for c, vv in v.categories.items():
                    create_dir(os.path.join(
                        args["de_models_dir"], "charts", "models", "results", dataset_name, c, f"total_by_{k}"))
                    [star_tasks.append((_plot_results_metrics, (k, models_conf, tmp, m, plot_args, "", (dataset_name, c))))
                        for m in TrainMetric.get_metrics()]

                    # print train results detailed at the CATEGORY granularity condensing plots by the parameters of interest
                    # (e.g., each box in the boxplot refers to a specific features F' and all models with that F'
                    # but other different parameters, such as all packets P)
                    if args["boxplot"] and sum(len(getattr(models_conf.train_params, kk))
                                               for kk in models_conf.train_params.train_combs() if kk != k) > 1:
                        create_dir(os.path.join(
                            args["de_models_dir"], "charts", "models", "results", dataset_name, c, f"condensed_by_{k}"))
                        [star_tasks.append((_plot_results_metrics_boxplot, (k, models_conf, args["de_models_dir"],
                                                                            m, "models", plot_args, (dataset_name, c))))
                         for m in TrainMetric.get_metrics()]

                    # print train results detailed at the PCAP granularity for each parameter
                    # (e.g., feature F in the x-axes, packets P in the x-axes)
                    for capture in vv.captures:
                        create_dir(os.path.join(
                            args["de_models_dir"], "charts", "models", "results", dataset_name, c, capture, f"total_by_{k}"))
                        [star_tasks.append((_plot_results_metrics,
                                            (k, models_conf, tmp, m, plot_args, "", (dataset_name, c, capture))))
                            for m in TrainMetric.get_metrics()]
                        # print train results detailed at the PCAP granularity condensing plots by the parameters of interest
                        # (e.g., each box in the boxplot refers to a specific features F' and all models with that F'
                        # but other different parameters, such as all packets P)
                        if args["boxplot"] and sum(len(getattr(models_conf.train_params, kk))
                                                   for kk in models_conf.train_params.train_combs() if kk != k) > 1:
                            create_dir(os.path.join(
                                args["de_models_dir"], "charts", "models", "results", dataset_name,
                                c, capture, f"condensed_by_{k}"))
                            [star_tasks.append((_plot_results_metrics_boxplot,
                                                (k, models_conf, args["de_models_dir"],
                                                 m, "models", plot_args, (dataset_name, c, capture))))
                             for m in TrainMetric.get_metrics()]

    if args["test_results"]:
        # look for folder with results of either one of the two test types (NORMAL and THROUGHPUT)
        for x in os.listdir(args["de_models_dir"]):
            tmp = os.path.join(args["de_models_dir"], x)
            if not os.path.isdir(tmp) or (x != "normal_test" and x != "throughput_test") or\
                    not os.path.isfile(os.path.join(tmp, "conf.json")):
                continue

            # print test results for each parameter (e.g., feature F in the x-axes, packets P in the x-axes)
            for k in models_conf.train_params.train_combs():
                if len(getattr(models_conf.train_params, k)) <= 1:
                    continue
                create_dir(os.path.join(
                    args["de_models_dir"], "charts", x, "results", f"total_by_{k}"))
                [star_tasks.append((_plot_results_metrics, (k, models_conf, tmp, m, plot_args)))
                    for m in TestMetric.get_metrics()]

                # print test results condensing plots by the parameters of interest
                # (e.g., each box in the boxplot refers to a specific features F' and all models with that F'
                # but other different parameters, such as all packets P)
                if args["boxplot"] and sum(len(getattr(models_conf.train_params, kk))
                                           for kk in models_conf.train_params.train_combs() if kk != k) > 1:
                    create_dir(os.path.join(
                        args["de_models_dir"], "charts", x, "results", f"condensed_by_{k}"))
                    [star_tasks.append((_plot_results_metrics_boxplot, (k, models_conf, args["de_models_dir"],
                                                                        m, x, plot_args)))
                     for m in TestMetric.get_metrics()]

    if args["test_results_detailed"]:
        # look for folder with results of either one of the two test types (NORMAL and THROUGHPUT)
        for x in os.listdir(args["de_models_dir"]):
            tmp = os.path.join(args["de_models_dir"], x)
            if not os.path.isdir(tmp) or (x != "normal_test" and x != "throughput_test") or\
                    not os.path.isfile(os.path.join(tmp, "conf.json")):
                continue

            # print test results for each parameter (e.g., feature F in the x-axes, packets P in the x-axes)
            for k in models_conf.train_params.train_combs():
                if len(getattr(models_conf.train_params, k)) <= 1:
                    continue
                for dataset_name, v in dataset_conf.online.datasets.items():
                    create_dir(os.path.join(
                        args["de_models_dir"], "charts", x, "results", dataset_name, f"total_by_{k}"))
                    [star_tasks.append((_plot_results_metrics, (k, models_conf, tmp, m, plot_args, "", (dataset_name,))))
                        for m in TestMetric.get_metrics()]

                    # print test results detailed at the DATASET granularity condensing plots by the parameters of interest
                    # (e.g., each box in the boxplot refers to a specific features F' and all models with that F'
                    # but other different parameters, such as all packets P)
                    if args["boxplot"] and sum(len(getattr(models_conf.train_params, kk))
                                               for kk in models_conf.train_params.train_combs() if kk != k) > 1:
                        create_dir(os.path.join(
                            args["de_models_dir"], "charts", x, "results", dataset_name, f"condensed_by_{k}"))
                        [star_tasks.append((_plot_results_metrics_boxplot, (k, models_conf, args["de_models_dir"],
                                                                            m, x, plot_args, (dataset_name,))))
                         for m in TestMetric.get_metrics()]

                    # print test results detailed at the CATEGORY granularity for each parameter
                    # (e.g., feature F in the x-axes, packets P in the x-axes)
                    for c, vv in v.categories.items():
                        create_dir(os.path.join(
                            args["de_models_dir"], "charts", x, "results", dataset_name, c, f"total_by_{k}"))
                        [star_tasks.append((_plot_results_metrics, (k, models_conf, tmp, m, plot_args, "", (dataset_name, c))))
                            for m in TestMetric.get_metrics()]

                        # print test results detailed at the CATEGORY granularity condensing plots by the parameters
                        # of interest (e.g., each box in the boxplot refers to a specific features F' and all models
                        # with that F' but other different parameters, such as all packets P)
                        if args["boxplot"] and sum(len(getattr(models_conf.train_params, kk))
                                                   for kk in models_conf.train_params.train_combs() if kk != k) > 1:
                            create_dir(os.path.join(
                                args["de_models_dir"], "charts", x, "results", dataset_name, c, f"condensed_by_{k}"))
                            [star_tasks.append((_plot_results_metrics_boxplot, (k, models_conf, args["de_models_dir"],
                                                                                m, x, plot_args, (dataset_name, c))))
                             for m in TestMetric.get_metrics()]

                        # print test results detailed at the PCAP granularity for each parameter
                        # (e.g., feature F in the x-axes, packets P in the x-axes)
                        for capture in vv.captures:
                            create_dir(os.path.join(
                                args["de_models_dir"], "charts", x, "results", dataset_name, c, capture, f"total_by_{k}"))
                            [star_tasks.append((_plot_results_metrics,
                                                (k, models_conf, tmp, m, plot_args, "", (dataset_name, c, capture))))
                                for m in TestMetric.get_metrics()]
                            # print test results detailed at the PCAP granularity condensing plots by the parameters
                            # of interest (e.g., each box in the boxplot refers to a specific features F' and
                            # all models with that F' but other different parameters, such as all packets P)
                            if args["boxplot"] and sum(len(getattr(models_conf.train_params, kk))
                                                       for kk in models_conf.train_params.train_combs() if kk != k) > 1:
                                create_dir(os.path.join(
                                    args["de_models_dir"], "charts", x, "results", dataset_name,
                                    c, capture, f"condensed_by_{k}"))
                                [star_tasks.append((_plot_results_metrics_boxplot,
                                                    (k, models_conf, args["de_models_dir"],
                                                     m, x, plot_args, (dataset_name, c, capture))))
                                 for m in TestMetric.get_metrics()]

    if args["transfer_learning"]:
        # look for folder with results of either a transfer learning
        for x in os.listdir(args["de_models_dir"]):
            tmp = os.path.join(args["de_models_dir"], x)
            if not os.path.isdir(tmp) or not x.startswith("transfer"):
                continue

            # print test results for each parameter (e.g., feature F in the x-axes, packets P in the x-axes)
            for k in models_conf.train_params.train_combs():
                if len(getattr(models_conf.train_params, k)) <= 1:
                    continue
                create_dir(os.path.join(
                    args["de_models_dir"], "charts", x, "results", f"total_by_{k}"))
                [star_tasks.append((_plot_results_metrics, (k, models_conf, tmp, m, plot_args)))
                    for m in TrainMetric.get_metrics()]

                # print test results condensing plots by the parameters of interest
                # (e.g., each box in the boxplot refers to a specific features F' and all models with that F'
                # but other different parameters, such as all packets P)
                if args["boxplot"] and sum(len(getattr(models_conf.train_params, kk))
                                           for kk in models_conf.train_params.train_combs() if kk != k) > 1:
                    create_dir(os.path.join(
                        args["de_models_dir"], "charts", x, "results", f"condensed_by_{k}"))
                    [star_tasks.append((_plot_results_metrics_boxplot, (k, models_conf, args["de_models_dir"],
                                                                        m, x, plot_args)))
                     for m in TrainMetric.get_metrics()]

    if args["transfer_learning_detailed"]:
        # look for folder with results of either one of the two test types (NORMAL and THROUGHPUT)
        for x in os.listdir(args["de_models_dir"]):
            tmp = os.path.join(args["de_models_dir"], x)
            if not os.path.isdir(tmp) or not x.startswith("transfer"):
                continue

            other_conf = DatasetConfig(
                **load_json_data(os.path.join(tmp, "conf.json")))
            # print test results for each parameter (e.g., feature F in the x-axes, packets P in the x-axes)
            for k in models_conf.train_params.train_combs():
                if len(getattr(models_conf.train_params, k)) <= 1:
                    continue

                for dataset_name, v in other_conf.offline.datasets.items():
                    create_dir(os.path.join(
                        args["de_models_dir"], "charts", x, "results", dataset_name, f"total_by_{k}"))
                    [star_tasks.append((_plot_results_metrics, (k, models_conf, tmp, m, plot_args, "", (dataset_name,))))
                        for m in TrainMetric.get_metrics()]

                    # print test results detailed at the DATASET granularity condensing plots by the parameters of interest
                    # (e.g., each box in the boxplot refers to a specific features F' and all models with that F'
                    # but other different parameters, such as all packets P)
                    if args["boxplot"] and sum(len(getattr(models_conf.train_params, kk))
                                               for kk in models_conf.train_params.train_combs() if kk != k) > 1:
                        create_dir(os.path.join(
                            args["de_models_dir"], "charts", x, "results", dataset_name, f"condensed_by_{k}"))
                        [star_tasks.append((_plot_results_metrics_boxplot, (k, models_conf, args["de_models_dir"],
                                                                            m, x, plot_args, (dataset_name,))))
                         for m in TrainMetric.get_metrics()]

                    # print test results detailed at the CATEGORY granularity for each parameter
                    # (e.g., feature F in the x-axes, packets P in the x-axes)
                    for c, vv in v.categories.items():
                        create_dir(os.path.join(
                            args["de_models_dir"], "charts", x, "results", dataset_name, c, f"total_by_{k}"))
                        [star_tasks.append((_plot_results_metrics, (k, models_conf, tmp, m, plot_args, "", (dataset_name, c))))
                            for m in TrainMetric.get_metrics()]

                        # print test results detailed at the CATEGORY granularity condensing plots by the parameters
                        # of interest (e.g., each box in the boxplot refers to a specific features F' and all models
                        # with that F' but other different parameters, such as all packets P)
                        if args["boxplot"] and sum(len(getattr(models_conf.train_params, kk))
                                                   for kk in models_conf.train_params.train_combs() if kk != k) > 1:
                            create_dir(os.path.join(
                                args["de_models_dir"], "charts", x, "results", dataset_name, c, f"condensed_by_{k}"))
                            [star_tasks.append((_plot_results_metrics_boxplot, (k, models_conf, args["de_models_dir"],
                                                                                m, x, plot_args, (dataset_name, c))))
                             for m in TrainMetric.get_metrics()]

                        # print test results detailed at the PCAP granularity for each parameter
                        # (e.g., feature F in the x-axes, packets P in the x-axes)
                        for capture in vv.captures:
                            create_dir(os.path.join(
                                args["de_models_dir"], "charts", x, "results", dataset_name, c, capture, f"total_by_{k}"))
                            [star_tasks.append((_plot_results_metrics,
                                                (k, models_conf, tmp, m, plot_args, "", (dataset_name, c, capture))))
                                for m in TrainMetric.get_metrics()]
                            # print test results detailed at the PCAP granularity condensing plots by the parameters
                            # of interest (e.g., each box in the boxplot refers to a specific features F' and
                            # all models with that F' but other different parameters, such as all packets P)
                            if args["boxplot"] and sum(len(getattr(models_conf.train_params, kk))
                                                       for kk in models_conf.train_params.train_combs() if kk != k) > 1:
                                create_dir(os.path.join(
                                    args["de_models_dir"], "charts", x, "results", dataset_name,
                                    c, capture, f"condensed_by_{k}"))
                                [star_tasks.append((_plot_results_metrics_boxplot,
                                                    (k, models_conf, args["de_models_dir"],
                                                     m, x, plot_args, (dataset_name, c, capture))))
                                 for m in TrainMetric.get_metrics()]

    with multiprocessing.Pool(maxtasksperchild=1) as pool:
        pool.starmap(handler, star_tasks)
