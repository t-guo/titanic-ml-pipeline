import logging
import luigi
import os
from itertools import compress
from sklearn.feature_selection import SelectFromModel

import pipeline.utils as utils
from luigi_extension import ConfigurableTask
from pipeline.data_transformation import TitanicFeatureTransformer
from pipeline.tasks.build_tasks import LogBuildName


LOGGER = logging.getLogger('luigi-interface')


class FeatureProcess(ConfigurableTask):

    def requires(self):
        return {
            "log_name": LogBuildName()
        }

    def output(self):
        return {
            "X_train": luigi.LocalTarget(os.path.join(self.model["data_repository"], "ProcessedFeatures",
                                                      "processed_train_features.pickle")),
            "y": luigi.LocalTarget(os.path.join(self.model["data_repository"], "ProcessedFeatures",
                                                "processed_targets.pickle")),
            "X_test": luigi.LocalTarget(os.path.join(self.model["data_repository"], "ProcessedFeatures",
                                                     "processed_test_features.pickle")),
            "transformer": luigi.LocalTarget(os.path.join(self.model["data_repository"], "ProcessedFeatures",
                                                          "transformer.pickle")),
        }

    def run(self):
        [utils.create_folder(self.output()[x].path) for x in self.output().keys()]

        # Load data
        training_data = utils.load_data("data/train.csv")
        test_data = utils.load_data("data/test.csv")

        # Combine test and train for data transformation
        # This may not be the best strategy in many real world applications as we are imputing our data with information
        # from test, but for the purpose of improving kaggle score, let's add test here for a better imputation
        combined = training_data.append(test_data, ignore_index=True)
        combined.drop(columns=["Survived"], inplace=True)

        # Fit and transform raw features
        LOGGER.info('{}: Transforming raw features'.format(repr(self)))
        transformer = TitanicFeatureTransformer()
        combined_transformed = transformer.fit_transform(combined)

        # Split back into train and test features
        X_train = combined_transformed[:len(training_data)]
        X_test = combined_transformed[len(training_data):]
        y = training_data["Survived"]

        # Save
        LOGGER.info('{}: Saving transformer and transformed features. {} rows of train data, '
                    '{} rows of test data, {} columns.'.format(repr(self), str(len(X_train)), str(len(X_test)), str(X_train.shape[1])))
        utils.save_data(transformer, self.output()["transformer"].path)
        utils.save_data(X_train, self.output()["X_train"].path)
        utils.save_data(X_test, self.output()["X_test"].path)
        utils.save_data(y, self.output()["y"].path)


class FeatureSelection(ConfigurableTask):

    def requires(self):
        return {
            "prepare_features": FeatureProcess()
        }

    def output(self):
        return {
            "X_train_filtered": luigi.LocalTarget(os.path.join(self.model["data_repository"], "SelectedFeatures",
                                                               "selected_train_features.pickle")),
            "X_test_filtered": luigi.LocalTarget(os.path.join(self.model["data_repository"], "SelectedFeatures",
                                                              "selected_test_features.pickle")),
            "feature_selection": luigi.LocalTarget(os.path.join(self.model["data_repository"], "SelectedFeatures",
                                                                "FeatureSelection.pickle")),
        }

    def run(self):
        [utils.create_folder(self.output()[x].path) for x in self.output().keys()]

        # Load data
        X_train = utils.load_data(self.input()["prepare_features"]["X_train"].path)
        X_test = utils.load_data(self.input()["prepare_features"]["X_test"].path)
        y = utils.load_data(self.input()["prepare_features"]["y"].path)
        transformer = utils.load_data(self.input()["prepare_features"]["transformer"].path)

        # Feature selection
        LOGGER.info('{}: Selecting features'.format(repr(self)))
        feature_selection_clf = utils.import_object(self.model["feature_selection"]["estimator"])()
        feature_selection_clf.set_params(**self.model["feature_selection"]["parameter_values"])

        feature_selection_model = SelectFromModel(feature_selection_clf)
        feature_selection_model.fit(X_train, y)

        X_train_filtered = feature_selection_model.transform(X_train)
        X_test_filtered = feature_selection_model.transform(X_test)

        # Save
        LOGGER.info('{}: Saving feature selection model and selected features. {} columns of data '
                    'selected.'.format(repr(self), str(X_train_filtered.shape[1])))

        columns = transformer.get_column_order()
        dropped_columns = feature_selection_model.get_support()
        columns_selected = list(compress(columns, list(dropped_columns)))
        LOGGER.info('{}: Selected columns: {}'.format(repr(self), ",".join(columns_selected)))

        utils.save_data(X_train_filtered, self.output()["X_train_filtered"].path)
        utils.save_data(X_test_filtered, self.output()["X_test_filtered"].path)
        utils.save_data(feature_selection_model, self.output()["feature_selection"].path)
