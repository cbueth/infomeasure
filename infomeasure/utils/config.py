"""Utility module for configuration settings."""

import logging

# Get the logger for this module with NullHandler
logging.getLogger("infomeasure").addHandler(logging.NullHandler())
logging.basicConfig(
    format="%(asctime)s | %(levelname)8s | %(filename)s:%(lineno)d | %(message)s",
    level=logging.INFO,
)
# set standard logger to 'infomeasure' logger
logger = logging.getLogger("infomeasure")


class Config:
    """Configuration settings for the package.

    This class provides configuration settings for the package. The settings are
    stored as class attributes and can be accessed and modified using the class
    methods.

    Default settings:

    - 'base': 2 (bits/shannons)
    - 'p_value_method': 'permutation_test'

    Attributes
    ----------
    _settings : dict
        A dictionary containing the configuration settings.

    """

    __default_settings = {
        "base": {
            "value": 2,  # 2: bits/shannon, e: nats, 10: hartleys/bans/dits
            "types": int | float,
            "additionally_allowed": ["e"],
        },
        "p_value_method": {
            "value": "permutation_test",
            "types": None,
            "additionally_allowed": ["permutation_test", "bootstrap"],
        },
    }
    _settings = {key: value["value"] for key, value in __default_settings.items()}

    @classmethod
    def get(cls, key: str):
        """Get the value of a configuration setting.

        Parameters
        ----------
        key : str
            The key of the configuration setting.

        Returns
        -------
        Any
            The value of the configuration setting.

        """
        return cls._settings[key]

    @classmethod
    def set(cls, key: str, value):
        """Set the value of a configuration setting.

        Parameters
        ----------
        key : str
            The key of the configuration setting.
        value : Any
            The value to set the configuration setting to.

        Raises
        ------
        KeyError
            If the key is not recognized.
        TypeError
            If the value is not of the correct type.
        """
        if key not in cls._settings:
            raise KeyError(f"Unknown configuration setting: {key}")
        if (
            cls.__default_settings[key]["types"] is None
            or not isinstance(value, cls.__default_settings[key]["types"])
        ) and (
            "additionally_allowed" not in cls.__default_settings[key]
            or value not in cls.__default_settings[key]["additionally_allowed"]
        ):
            raise TypeError(
                f"Invalid value '{value}' ({type(value)}) for setting '{key}'. "
                f"Expected type: {cls.__default_settings[key]['types']}"
                + (
                    f" or one of {cls.__default_settings[key]['additionally_allowed']}"
                    if "additionally_allowed" in cls.__default_settings[key]
                    else ""
                )
            )
        cls._settings[key] = value

    @classmethod
    def reset(cls):
        """Reset the configuration settings to the default values."""
        cls._settings = {
            key: value["value"] for key, value in cls.__default_settings.items()
        }

    @classmethod
    def set_logarithmic_unit(cls, unit: str):
        """Set the base for the logarithmic unit.

        The base determines the logarithmic unit used for entropy calculations:

        - 'bits' or 'shannons' (base 2)
        - 'nats' (base e)
        - 'hartleys', 'bans', or 'dits' (base 10)

        Alternatively, you can set the base directly using the 'base' key,
        via :meth:`set`.

        Parameters
        ----------
        unit : str
            The logarithmic unit to set. Use 'bit(s)' or 'shannon(s)' for base 2,
            'nat(s)' for base e, and 'hartley(s)', 'ban(s)', or 'dit(s)' for base 10.

        Raises
        ------
        ValueError
            If the unit is not recognized.

        """
        if unit.lower() in ["bits", "bit", "shannons", "shannon"]:
            cls.set("base", 2)
        elif unit.lower() in ["nats", "nat"]:
            cls.set("base", "e")
        elif unit.lower() in ["hartleys", "hartley", "bans", "ban", "dits", "dit"]:
            cls.set("base", 10)
        else:
            raise ValueError(f"Unknown logarithmic unit: {unit}")

    @staticmethod
    def set_log_level(level: int | str) -> None:
        """Set the logging level for the package.

        Parameters
        ----------
        level : int | str
            The logging level. See the :mod:`logging` module for more information.

        Raises
        ------
        ValueError
            If the level is not a valid logging level.
        """
        # get logging representation of level
        level = level if isinstance(level, int) else getattr(logging, level.upper())
        logger.setLevel(level)
