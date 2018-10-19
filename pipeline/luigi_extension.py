from multiprocessing import Manager
import luigi


# superclass that defines withConfig() method
class ConfigurableTask(luigi.Task):

    @classmethod
    def set_configs(cls, model_config):
        # Manager.dict() provides a proxy object which can be shared between multiple luigi processes
        cls.model = Manager().dict(model_config)
