import logging
import luigi
import os
import pandas as pd

import pipeline.utils as utils
from luigi_extension import ConfigurableTask
from pipeline.tasks.training_tasks import EnsembleVotingClassifier, TuneModelParameters

LOGGER = logging.getLogger('luigi-interface')


class Predict(ConfigurableTask):

    def requires(self):
        return {
            "ensemble_clf": EnsembleVotingClassifier(),
            "cv": TuneModelParameters()
        }

    def output(self):
        return {
            "predictions": luigi.LocalTarget(os.path.join(self.model["data_repository"], "Predictions",
                                                          "__predictions_made__.txt"))
        }

    def run(self):
        [utils.create_folder(self.output()[x].path) for x in self.output().keys()]

        # Read prediction data
        pred_data = utils.load_data("data/test.csv")

        predict_folder = os.path.dirname(self.output()["predictions"].path)

        # Read model and transform data
        final_model = utils.load_data(self.input()["ensemble_clf"]["ensemble_model"].path)
        transformer = final_model["transformer"]
        X_pred_transformed = transformer.transform(pred_data)

        best_model = utils.load_data(self.input()["cv"]["model_package"].path)
        for m in best_model:
            clf = m["best_model"]
            prediction_df = self.make_prediction(clf, X_pred_transformed, pred_data["PassengerId"])
            utils.save_data(prediction_df,
                            os.path.join(predict_folder,
                                         m["model"]["estimator_type"] + "_" + str(m["best_score"]) + ".csv"))

        eclf = final_model["final_model"]
        prediction_df = self.make_prediction(eclf, X_pred_transformed, pred_data["PassengerId"])
        utils.save_data(prediction_df, os.path.join(predict_folder, "ensemble.csv"))

        utils.save_data("", self.output()["predictions"].path)

    @staticmethod
    def make_prediction(clf, X, ids):
        y_pred = clf.predict(X)
        pred_df = pd.DataFrame(ids)
        pred_df["Survived"] = y_pred

        return pred_df