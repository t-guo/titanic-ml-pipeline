import hashlib
import json
import os
import logging
import luigi

import pipeline.utils as utils
from luigi_extension import ConfigurableTask

from pipeline.tasks.data_process_tasks import FeatureProcess
from pipeline.tasks.training_tasks import TuneModelParameters, EnsembleVotingClassifier
from pipeline.tasks.prediction_tasks import Predict


# config locations
MODEL_CONFIGS_DIR = utils.absolute_path_from_project_root(os.path.join('config.yaml'))
CONFIG_HASH_LENGTH = 9
LOGGER_NAME = 'luigi-interface'
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(module)s - %(message)s'
PROCESS_LOG_FILE_NAME = "training_process.log"
LOGGER = logging.getLogger(LOGGER_NAME)


def create_task_list():
    tasks = []
    tasks.append(FeatureProcess())
    tasks.append(TuneModelParameters())
    tasks.append(EnsembleVotingClassifier())
    tasks.append(Predict())

    return tasks


def get_job_configuration():
    config_path = MODEL_CONFIGS_DIR
    model_config = utils.load_yaml_config(config_path)
    config_string = json.dumps(model_config)
    model_config['config_id'] = hashlib.sha256(config_string).hexdigest()[:CONFIG_HASH_LENGTH]

    return model_config


def make_data_repository(data_repository, config_id):
    absolute_data_repository = utils.absolute_path_from_project_root(data_repository)
    data_repo_path = os.path.join(absolute_data_repository, config_id)
    if not os.path.exists(data_repo_path):
        utils.create_folder(os.path.join(data_repo_path))

    return data_repo_path


def main():
    model_config = get_job_configuration()

    print "Config ID: {}".format(model_config["config_id"])

    # initialize data repository
    model_config["data_repository"] = make_data_repository(
        model_config["data_repository"], model_config["config_id"])

    # configure logger
    log_file = os.path.join(model_config["data_repository"], PROCESS_LOG_FILE_NAME)

    # luigi overrides log level during setup and adds its own handler
    formatter = logging.Formatter(fmt=LOG_FORMAT)
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger = logging.getLogger(LOGGER_NAME)
    logger.addHandler(file_handler)
    logger.propagate = False

    LOGGER.info('Config ID: {}'.format(model_config['config_id']))
    LOGGER.info('Interim Directory: {}'.format(model_config["data_repository"]))

    # execute pipeline
    ConfigurableTask.set_configs(model_config)
    tasks = create_task_list()
    luigi.build(
        tasks,
        local_scheduler=model_config["local_scheduler"],
        workers=model_config["luigi_worker_count"],
        log_level=model_config["log_level"])

    success = all([task.complete() for task in tasks])
    return success


if __name__ == "__main__":
    main()
