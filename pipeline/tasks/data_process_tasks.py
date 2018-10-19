import logging
import luigi
import os

import pipeline.utils as utils
from luigi_extension import ConfigurableTask
from pipeline.titanic_data_transformation import TitanicFeatureTransformer


LOGGER = logging.getLogger('luigi-interface')


class FeatureProcess(ConfigurableTask):

    def output(self):
        return {
            "X": luigi.LocalTarget(os.path.join(self.model["data_repository"], "ProcessedFeatures",
                                                "processed_features.pickle")),
            "y": luigi.LocalTarget(os.path.join(self.model["data_repository"], "ProcessedFeatures",
                                                "processed_targets.pickle")),
            "transformer": luigi.LocalTarget(os.path.join(self.model["data_repository"], "ProcessedFeatures",
                                                          "transformer.pickle")),
        }

    def run(self):
        [utils.create_folder(self.output()[x].path) for x in self.output().keys()]

        # Load pipeline data
        training_data = utils.load_data("data/train.csv")
        X = training_data[training_data.columns.difference(['Survived'])]
        y = training_data["Survived"]

        # Fit and transform raw features
        LOGGER.info('{}: Transforming raw features'.format(repr(self)))
        transformer = TitanicFeatureTransformer()
        X_transformed = transformer.transform(X)

        # Save
        LOGGER.info('{}: Saving transformer and transformed features. {} rows of data.'.format(repr(self),
                                                                                               str(len(X_transformed))))
        utils.save_data(transformer, self.output()["transformer"].path)
        utils.save_data(X_transformed, self.output()["X"].path)
        utils.save_data(y, self.output()["y"].path)
