import yaml

from collections import namedtuple

CONFIG = './config.yml'

class Config:
    """
    A class used to provide configuration data (singleton).

    Class attributes
    ----------------
    config : namedtuple
        a named tuple containing all key value pairs provided by
        the configuration file config.yml

    Class methods
    -------------
    get_config(cls)
        Returns the content of the configuration file config.yml as
        named tuple.
    """
    config = None

    @classmethod
    def get_config(cls):
        """
        Reads the configuration, if not already done, and 
        returns it as named tuple.
        
        Returns:
        --------
        namedtuple:
            The content of config.yml as named tuple.
        """
        if Config.config is None:
            with open(CONFIG) as f:
                cf_dict = yaml.safe_load(f)
                Config.config = namedtuple(
                    'Config', cf_dict.keys())._make(cf_dict.values())
        return Config.config