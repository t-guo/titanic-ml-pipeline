import logging
import luigi
import os
import pandas as pd

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import VotingClassifier

import pipeline.utils as utils
from luigi_extension import ConfigurableTask
from pipeline.tasks.data_process_tasks import FeatureProcess

LOGGER = logging.getLogger('luigi-interface')


class TuneModelParameters(ConfigurableTask):
    def __init__(self):
        self.best_estimator_per_model = []
        super(TuneModelParameters, self).__init__()

    def requires(self):
        return {
            'prepare_features': FeatureProcess()
        }

    def output(self):
        return {
            'model_package': luigi.LocalTarget(os.path.join(self.model['data_repository'], 'CrossValidation',
                                                            'best_models.pickle'))
        }

    def run(self):

        [utils.create_folder(self.output()[x].path) for x in self.output().keys()]

        # Read in X and y
        X_train = utils.load_data(self.input()['prepare_features']['X'].path)
        y_train = utils.load_data(self.input()['prepare_features']['y'].path)

        # Iterate over our models and our offset targets, performing the grid search to tune hyper-parameters
        for model in self.model['sklearn_estimators']:
            LOGGER.info('{}: Tuning model - {}'.format(repr(self), model["estimator"]))

            grid_search = self.do_grid_search(model, X_train, y_train)
            self.best_model_per_model_type(model, grid_search)

        # Save best models
        for best_model in self.best_estimator_per_model:
            LOGGER.info('{}: BEST {} - {}'.format(repr(self),
                                                  best_model["model"]["estimator_type"],
                                                  str(best_model["best_score"])))

        utils.save_data(self.best_estimator_per_model, self.output()['model_package'].path)

    def do_grid_search(self, model_config, X_train, y_train):
        # Import and instantiate classifier specified in model config
        clf = utils.import_object(model_config['estimator'])()

        # Pull out parameters and names
        param_grid = model_config['parameter_values']

        cv = StratifiedKFold(n_splits=self.model["n_folds"])
        grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=cv, n_jobs=1, refit=True,
                                   scoring=model_config["scorer"], verbose=4, return_train_score=True, iid=False)

        grid_search.fit(X_train, y_train)

        return grid_search

    def best_model_per_model_type(self, model_config, grid_search):
        cv_best_model = grid_search.best_estimator_
        cv_best_score = grid_search.best_score_

        self.best_estimator_per_model.append({'model': model_config, 'best_model': cv_best_model,
                                              'best_score': cv_best_score})


class EnsembleVotingClassifier(ConfigurableTask):

    def requires(self):
        return {
            'prepare_features': FeatureProcess(),
            'cv': TuneModelParameters()
        }

    def output(self):
        return {
            "ensemble_model": luigi.LocalTarget(os.path.join(self.model["data_repository"], "FinalModel",
                                                             "final_model.pickle"))
        }

    def run(self):
        [utils.create_folder(self.output()[x].path) for x in self.output().keys()]

        # Read best models
        best_models = utils.load_data(self.input()["cv"]["model_package"].path)
        transformer = utils.load_data(self.input()["prepare_features"]["transformer"].path)

        # Read in X and y
        X_train = utils.load_data(self.input()['prepare_features']['X'].path)
        y_train = utils.load_data(self.input()['prepare_features']['y'].path)

        estimators = []
        for model in best_models:
            estimators.append((model["model"]["estimator_type"], model["best_model"]))

        eclf = VotingClassifier(estimators=estimators, voting="soft")

        LOGGER.info('{}: Fitting ensemble model '.format(repr(self)))
        eclf.fit(X_train, y_train)

        # Package model
        final_model_package = {
            "transformer": transformer,
            "final_model": eclf
        }

        # Save ensemble model
        utils.save_data(final_model_package, self.output()["ensemble_model"].path)
