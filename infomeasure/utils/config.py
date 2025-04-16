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

# Define a dictionary to map base values to their corresponding unit names
BASE_UNIT_MAP = {
    2: {
        "identifiers": ["bits", "bit", "shannons", "shannon"],
        "name": "bit/shannon",
        "description": "This base is best suited for binary information, "
        "which is fundamental in digital systems.",
    },
    "e": {
        "identifiers": ["nats", "nat", "nit", "nepit"],
        "name": "nat/nit/nepit",
        "description": "This base is best suited for "
        "continuous probability distributions.",
    },
    10: {
        "identifiers": ["hartleys", "hartley", "bans", "ban", "dits", "dit"],
        "name": "hartley/ban/dit",
        "description": "This base is best suited for decimal information, "
        "which is commonly used in human-centric measurements.",
    },
}


class Config:
    """Configuration settings for the package.

    This class provides configuration settings for the package. The settings are
    stored as class attributes and can be accessed and modified using the class
    methods.

    Default settings:

    - ``base``: "e" (nats)
    - ``p_value_method``: "permutation_test"

    Attributes
    ----------
    _settings : dict
        A dictionary containing the configuration settings.

    """

    __default_settings = {
        "base": {
            "value": "e",  # 2: bits/shannon, e: nats, 10: hartleys/bans/dits
            "types": int | float,
            "explicitly_allowed": ["e"],
        },
        "p_value_method": {
            "value": "permutation_test",
            "types": None,
            "explicitly_allowed": ["permutation_test", "bootstrap"],
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
            "explicitly_allowed" not in cls.__default_settings[key]
            or value not in cls.__default_settings[key]["explicitly_allowed"]
        ):
            raise TypeError(
                f"Invalid value '{value}' ({type(value)}) for setting '{key}'. "
                f"Expected type: {cls.__default_settings[key]['types']}"
                + (
                    f" or one of {cls.__default_settings[key]['explicitly_allowed']}"
                    if "explicitly_allowed" in cls.__default_settings[key]
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
        unit = unit.lower()
        for base, units in BASE_UNIT_MAP.items():
            if unit in units["identifiers"]:
                cls.set("base", base)
                return
        raise ValueError(f"Unknown logarithmic unit: {unit}")

    @classmethod
    def get_logarithmic_unit(cls) -> str:
        """Get the logarithmic unit for entropy calculations.

        Returns
        -------
        str
            The logarithmic unit.
        """
        for base, units in BASE_UNIT_MAP.items():
            if cls.get("base") == base:
                return units["name"]
        return f"unknown (base {cls.get('base')})"

    @classmethod
    def get_logarithmic_unit_description(cls) -> str:
        """Get the description of the logarithmic unit for entropy calculations.

        Returns
        -------
        str
            The description of the logarithmic unit.

        Raises
        ------
        ValueError
            If there is no description for the logarithmic unit.
        """
        for base, units in BASE_UNIT_MAP.items():
            if cls.get("base") == base:
                return units["description"]
        raise ValueError(f"No description for logarithmic unit: base {cls.get('base')}")

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


logger.debug(
    "Using %s (base %s) for entropy calculations.\n%s\n"
    r"Use 'infomeasure.Config.set_logarithmic_unit(unit)' to change the unit or "
    r"'infomeasure.Config.set(base)' to set the base directly.",
    Config.get_logarithmic_unit(),
    Config.get("base"),
    Config.get_logarithmic_unit_description(),
)
