"""Test configuration class for the infomeasure module."""

import pytest

from infomeasure.utils import Config


class TestConfig:
    """Test configuration class for the infomeasure module."""

    def test_get(self):
        """Test the get method of the Config class."""
        assert Config.get("base") == "e"

    def test_set(self):
        """Test the set method of the Config class."""
        Config.set("base", 10)
        assert Config.get("base") == 10
        Config.set("base", 2)
        assert Config.get("base") == 2
        Config.set("base", "e")
        assert Config.get("base") == "e"

    def test_set_invalid_key(self):
        """Test setting an invalid key."""
        with pytest.raises(KeyError):
            Config.set("invalid_key", 10)

    def test_get_invalid_key(self):
        """Test getting an invalid key."""
        with pytest.raises(KeyError):
            Config.get("invalid_key")

    def test_set_invalid_value(self):
        """Test setting an invalid value."""
        with pytest.raises(TypeError):
            Config.set("base", "invalid_value")

    @pytest.mark.parametrize(
        "unit, expected_base",
        [
            ("bits", 2),
            ("dits", 10),
            ("shannon", 2),
            ("shannons", 2),
            ("nats", "e"),
            ("hartley", 10),
            ("nat", "e"),
            ("hartleys", 10),
            ("bans", 10),
            ("ban", 10),
            ("bit", 2),
            ("dit", 10),
        ],
    )
    def test_set_get_logarithmic_unit(self, unit, expected_base):
        """Test the set_logarithmic_unit method of the Config class."""
        Config.set_logarithmic_unit(unit)
        assert Config.get("base") == expected_base
        assert (
            unit[:-1] if unit.endswith("s") else unit
        ) in Config.get_logarithmic_unit()
        Config.get_logarithmic_unit_description()

    def test_get_unknown_unit(self):
        """Test getting an unknown logarithmic unit and description."""
        Config.set("base", 3)
        assert Config.get_logarithmic_unit() == "unknown (base 3)"
        with pytest.raises(
            ValueError, match="No description for logarithmic unit: base 3"
        ):
            Config.get_logarithmic_unit_description()

    def test_set_logarithmic_unit_invalid(self):
        """Test setting an invalid logarithmic unit."""
        with pytest.raises(ValueError):
            Config.set_logarithmic_unit("invalid_unit")

    def test_reset(self):
        """Test the reset method of the Config class."""
        Config.set("base", 10)
        Config.reset()
        assert Config.get("base") == "e"
