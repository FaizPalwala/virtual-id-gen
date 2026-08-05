import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from generate_identities import build_variation_plan


def test_default_variation_plan_has_hundred_unique_variants():
    plan = build_variation_plan()
    assert len(plan) == 100
    assert [item.variation_id for item in plan] == list(range(100))
    assert len({item.prompt() for item in plan}) == 100


def test_variations_do_not_explicitly_change_identity_traits():
    forbidden = ("ethnicity", "race", r"\bage\b", "hair colour", "gender")
    prompts = " ".join(item.prompt().lower() for item in build_variation_plan())
    assert not any(re.search(word, prompts) for word in forbidden)
