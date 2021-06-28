from yaml import load as yaml_load, FullLoader as yaml_FullLoader


class Configurations(object):
    """
    Simple class to freeze the attributes once they are created in the object.
    """
    def __setattr__(self, key, value):
        if key not in self.__dict__:
            object.__setattr__(self, key, value)
        else:
            raise ValueError(f"Cannot set `{key}` twice. Attribute `{key}` value is already set in {self} as `{self.__dict__[key]}`.")


def dict_to_object(_dict):
    """
    Input dictionary is converted to an object with all the dictionary keys as its attribute.

    Args:
        _dict (dict): Dictionary.

    Returns:
        Configurations: Object with all the key value pairs in the input dictionary as the attributes and their values.
    """

    __obj = Configurations()

    for k, v in _dict.items():
        if type(v) is dict:
            v = dict_to_object(v)
        if not hasattr(__obj, k):
            setattr(__obj, k, v)

    return __obj


def yaml_to_object(yaml_file):
    """
    Read a yaml file and convert it to a python object.

    Args:
        yaml_file (str): Path to ``.yaml`` file.

    Returns:
        Configurations: Object with all the keys in the given ``.yaml`` file as its attributes.
    """
    with open(yaml_file) as f:
        config = yaml_load(f, Loader=yaml_FullLoader)
    return dict_to_object(config)


if __name__ == "__main__":
    config = yaml_to_object("./../config.yaml")

    print(config.data.MPII.path.base)
    try:
        config.data.MPII.path.base = "MyPath"
    except ValueError as e:
        print(e)
        print(f"Cannot change frozen object {config}")
        print("Test successful!")

    print("Done.")
