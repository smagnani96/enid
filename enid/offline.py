import argparse
import os

from .datasetter import DatasetConfig
from .lib.utility import dump_json_data, get_logger, load_json_data, set_seed
from .trainer import ModelConfig, ResultTrain

_logger = get_logger(__name__)


def main(args_list):
    # registering cli parameters, which can be shown with the -h
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        'de_models_dir', help='path to the model directory containing the models to test')
    parser.add_argument(
        '-r', '--retest', help='overwrite previous result', action="store_true")
    args = parser.parse_args(args_list).__dict__

    models_dir = os.path.join(
        args["de_models_dir"], 'models')

    models_config = ModelConfig(
        **load_json_data(os.path.join(models_dir, "conf.json")))
    dataset_config = DatasetConfig(
        **load_json_data(os.path.join(args["de_models_dir"], os.pardir, "conf.json")))

    for c in models_config.train_params.all_train_combs():
        name = models_config.detection_engine.model_name(**c)
        new_path = os.path.join(models_dir, name)
        if not args["retest"] and os.path.isfile(os.path.join(new_path, "results.json")):
            _logger.info(
                f"Offline results for {name} already present, skipping")
            continue

        params = load_json_data(os.path.join(new_path, "params.json"))

        fholder, params, model = models_config.detection_engine.load_model(
            params, os.path.join(args["de_models_dir"], "models"))
        set_seed()
        (_, _, _), (_, _, _), (xts, yts, pt) = models_config.detection_engine._load_dataset(
            os.path.join(args["de_models_dir"], os.pardir), fholder, **params.__dict__)

        _logger.info(f"Testing {name}")
        y_pred = models_config.detection_engine.predict(
            model, xts, **params.__dict__)

        res = ResultTrain(
            _threshold=models_config.train_params.malicious_threshold,
            _ypred=y_pred, _ytrue=yts)
        res.update(dataset_config.offline, pt)
        dump_json_data(res, os.path.join(new_path, "results.json"))
