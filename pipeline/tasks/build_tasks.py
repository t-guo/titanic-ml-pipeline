import logging
import luigi
import os

import pipeline.utils as utils
from luigi_extension import ConfigurableTask


LOGGER = logging.getLogger('luigi-interface')


class LogBuildName(ConfigurableTask):

    def output(self):
        return {
            "log_name": luigi.LocalTarget(os.path.join(self.model["data_repository"], "build_description.txt")),
        }

    def run(self):
        [utils.create_folder(self.output()[x].path) for x in self.output().keys()]

        utils.save_data(self.model["build_description"], self.output()["log_name"].path)
