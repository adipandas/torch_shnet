import os.path as osp
import yaml


class ConfigYamlParserMPII:
    """
    Reads and Parses the ``config.yaml`` file for the project.

    Args:
        path (str): ``config.yaml`` file path.
    """
    def __init__(self, path="./config.yaml"):

        self.config_file_path = osp.abspath(path)
        assert osp.exists(self.config_file_path), f"File not found '{self.config_file_path}'."

        with open(self.config_file_path) as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)

        self._set_dataset_directories()
        self._set_dataset_parameters()
        self._set_neural_network_parameters()

    def _set_dataset_directories(self):
        _paths = self.config['data']['MPII']['paths']

        self.BASE_DIR = osp.join(osp.dirname(self.config_file_path), _paths['base'])
        self._exists(self.BASE_DIR)

        self.IMAGE_DIR = osp.join(self.BASE_DIR, _paths['images'])
        self._exists(self.IMAGE_DIR)

        self.TRAINING_ANNOTATION_FILE = osp.join(self.BASE_DIR, _paths['annotations']['training'])
        self._exists(self.TRAINING_ANNOTATION_FILE)

        self.VALIDATION_ANNOTATION_FILE = osp.join(self.BASE_DIR, _paths['annotations']['validation'])
        self._exists(self.VALIDATION_ANNOTATION_FILE)

        self.TESTING_ANNOTATION_FILE = osp.join(self.BASE_DIR, _paths['annotations']['testing'])
        self._exists(self.TESTING_ANNOTATION_FILE)

    def _set_dataset_parameters(self):
        self.REFERENCE_IMAGE_SIZE = self.config['data']['MPII']['reference_image_size']
        self.PARTS = self.config['data']['MPII']['parts']
        self.PART_PAIRS = self.config['data']['MPII']['part_pairs']

    def _set_neural_network_parameters(self):
        _nn = self.config['neural_network']
        self.POSENET_INPUT_PARAMS = _nn['PoseNet']
        self.NN_TRAINING_PARAMS = _nn['train']
        self.NN_INFERENCE_PARAMS = _nn['inference']

    def _exists(self, path):
        """
        To check if the given path is valid

        Args:
            path (str): File or directory path.
        """
        assert osp.exists(path), f"'{path}' not found. Check '{self.config_file_path}'."


if __name__ == "__main__":
    c = ConfigYamlParserMPII(path="./../config.yaml")
    print(c.config)

