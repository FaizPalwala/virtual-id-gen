import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from instantid_adapter import validate_generation_settings


def test_generation_settings_accept_valid_sdxl_dimensions():
    validate_generation_settings(1024, 1024, 35, 0.9, 0.8)


def test_generation_settings_reject_invalid_resolution():
    try:
        validate_generation_settings(1025, 1024, 35, 0.9, 0.8)
    except ValueError as error:
        assert "divisible by 8" in str(error)
    else:
        raise AssertionError("Expected an invalid resolution error")
