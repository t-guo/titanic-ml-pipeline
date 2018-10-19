import logging
import luigi
import os
import pandas as pd

import pipeline.utils as utils
from luigi_extension import ConfigurableTask
from pipeline.tasks.training_tasks import EnsembleVotingClassifier

LOGGER = logging.getLogger('luigi-interface')


class Predict(ConfigurableTask):

    def requires(self):
        return {
            "ensemble_clf": EnsembleVotingClassifier()
        }

    def output(self):
        return {
            "predictions": luigi.LocalTarget(os.path.join(self.model["data_repository"], "Predictions",
                                                          "predictions.csv"))
        }

    def run(self):
        [utils.create_folder(self.output()[x].path) for x in self.output().keys()]

        # Read prediction data
        pred_data = utils.load_data("data/test.csv")

        # Read model and transformer
        final_model = utils.load_data(self.input()["ensemble_clf"]["ensemble_model"].path)
        transformer = final_model["transformer"]
        eclf = final_model["final_model"]

        # Transform and predict
        X_pred_transformed = transformer.transform(pred_data)
        y_pred = eclf.predict(X_pred_transformed)

        prediction_df = pd.DataFrame(pred_data["PassengerId"])
        prediction_df["Survived"] = y_pred

        utils.save_data(prediction_df, self.output()["predictions"].path)