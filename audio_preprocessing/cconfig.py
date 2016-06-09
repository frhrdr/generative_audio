import os


class Configuration(object):

    @property
    def datapath(self, dataset=None):
        return self.get_datapath()

    def get_datapath(self, dataset=None):
        if dataset is None:
            return os.environ.get("PYTHON_DATA_FOLDER", "data")
        env_variable = "PYTHON_DATA_FOLDER_%s" % dataset.upper()
        return os.environ.get(env_variable, "data")

    @property
    def frequency_of_format(self):
        return 44100   # frequency of file format, in this case wav files

    def get_frequency_file(self):
        return os.environ.get("PYTHON_FREQ_FILENAME", "UNUSED")


config = Configuration()
