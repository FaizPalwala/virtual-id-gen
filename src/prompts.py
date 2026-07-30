"""Prompt templates and variation plans for identity-conditioned generation.

This module owns all prompt-related data: the photorealism prefix, the
negative prompt, the ``VariationSpec`` dataclass, and the function that
builds deterministic variation plans from composition and lighting grids.

Prompts are pure data — no model loading, no I/O, no dependencies beyond
stdlib.  Import and use from ``generate_identities.py`` (or directly when
debugging prompt diversity)::

    from prompts import build_variation_plan, VariationSpec

    plan = build_variation_plan(85)
    for spec in plan[:3]:
        print(spec.prompt())
"""

from __future__ import annotations

from dataclasses import dataclass

PHOTOREALISM_PREFIX = (
    "RAW photo, photorealistic DSLR portrait of the same person, realistic "
    "skin texture and pores, realistic hair strands, natural colour grading, "
    "sharp eyes, high detail"
)

NEGATIVE_PROMPT = (
    "painting, illustration, drawing, CGI, 3D render, cartoon, anime, "
    "monochrome, grayscale, sepia, desaturated, waxy skin, airbrushed skin, "
    "lowres, blurry, out of focus, distorted face, malformed face, extra face, "
    "duplicate face, text, watermark, logo"
)


@dataclass(frozen=True)
class VariationSpec:
    """A non-identity rendering instruction for one generated candidate."""

    variation_id: int
    pose: str
    expression: str
    lighting: str
    setting: str
    camera: str

    def prompt(self) -> str:
        """Render the structured plan as a prompt without changing identity traits."""
        return ", ".join(
            (
                PHOTOREALISM_PREFIX,
                self.pose,
                self.expression,
                self.lighting,
                self.setting,
                self.camera,
            )
        )


def build_variation_plan(count: int = 100) -> list[VariationSpec]:
    """Return deterministic, diverse non-identity prompts.

    The first 100 entries form a balanced 20 x 5 design: 20 pose/expression/
    setting/composition tuples crossed with 5 lighting treatments.  Counts above
    100 repeat the composition plan with an additional deterministic lighting
    cycle; this supports candidate oversampling without a fixed portrait prompt.
    """
    if count <= 0:
        raise ValueError("variantsperidentity must be positive.")
    compositions = [
        (
            "frontal head-and-shoulders pose",
            "neutral relaxed expression",
            "clean dark-blue studio backdrop",
            "85mm portrait lens",
        ),
        (
            "frontal head-and-shoulders pose",
            "gentle closed-mouth smile",
            "clean neutral studio backdrop",
            "85mm portrait lens",
        ),
        (
            "three-quarter view facing left",
            "neutral relaxed expression",
            "softly blurred indoor background",
            "85mm portrait lens",
        ),
        (
            "three-quarter view facing right",
            "gentle closed-mouth smile",
            "softly blurred indoor background",
            "85mm portrait lens",
        ),
        (
            "slight head turn to the left",
            "calm thoughtful expression",
            "subtle dark studio backdrop",
            "85mm portrait lens",
        ),
        (
            "slight head turn to the right",
            "calm thoughtful expression",
            "subtle dark studio backdrop",
            "85mm portrait lens",
        ),
        (
            "upright seated head-and-shoulders pose",
            "neutral relaxed expression",
            "minimal professional office background",
            "85mm portrait lens",
        ),
        (
            "upright seated head-and-shoulders pose",
            "gentle closed-mouth smile",
            "minimal professional office background",
            "85mm portrait lens",
        ),
        (
            "natural candid head-and-shoulders pose",
            "soft relaxed expression",
            "softly blurred outdoor greenery",
            "85mm portrait lens",
        ),
        (
            "looking slightly above the camera",
            "neutral relaxed expression",
            "softly blurred outdoor shade",
            "85mm portrait lens",
        ),
        (
            "looking slightly past the camera",
            "calm thoughtful expression",
            "muted indoor background",
            "85mm portrait lens",
        ),
        (
            "straight-on close portrait",
            "subtle relaxed smile",
            "simple warm-grey studio backdrop",
            "85mm portrait lens",
        ),
        (
            "three-quarter editorial portrait",
            "confident neutral expression",
            "softly blurred editorial interior",
            "85mm portrait lens",
        ),
        (
            "profile view facing left",
            "neutral relaxed expression",
            "clean grey backdrop",
            "85mm portrait lens",
        ),
        (
            "profile view facing right",
            "gentle closed-mouth smile",
            "clean grey backdrop",
            "85mm portrait lens",
        ),
        (
            "head tilted slightly up",
            "curious expression",
            "softly blurred foliage background",
            "85mm portrait lens",
        ),
        (
            "looking over shoulder",
            "warm genuine smile",
            "muted urban brick wall background",
            "85mm portrait lens",
        ),
        (
            "extreme close-up portrait",
            "intense focused expression",
            "dark minimal backdrop",
            "85mm portrait lens",
        ),
        (
            "candid mid-shot portrait",
            "soft natural laugh",
            "outdoor cafe background",
            "50mm environmental lens",
        ),
        (
            "two-thirds standing portrait",
            "confident slight smirk",
            "architectural column backdrop",
            "35mm wide lens",
        ),
    ]
    lighting = [
        "soft natural window light",
        "soft professional studio key light with gentle fill",
        "open-shade daylight with balanced exposure",
        "warm golden hour sunlight from the side",
        "cool overcast window light",
    ]
    return [
        VariationSpec(
            variation_id=index,
            pose=compositions[(index // len(lighting)) % len(compositions)][0],
            expression=compositions[(index // len(lighting)) % len(compositions)][1],
            setting=compositions[(index // len(lighting)) % len(compositions)][2],
            camera=compositions[(index // len(lighting)) % len(compositions)][3],
            lighting=lighting[index % len(lighting)],
        )
        for index in range(count)
    ]
