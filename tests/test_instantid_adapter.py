import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from instantidadapter import build_parser, draw_keypoints, validate_arguments


def test_parser_accepts_pipeline_argument_contract():
    args = build_parser().parse_args([
        "--idimage", "seed.jpg", "--outputpath", "output.png",
        "--seed", "42", "--prompt", "test portrait",
        "--num-inference-steps", "5", "--guidance-scale", "4.5",
    ])
    assert args.idimage == "seed.jpg"
    assert args.outputpath == "output.png"
    assert args.num_inference_steps == 5
    assert args.guidance_scale == 4.5


def test_parser_accepts_legacy_underscore_aliases():
    args = build_parser().parse_args([
        "--id_image", "seed.jpg", "--output_path", "output.png",
        "--num_inference_steps", "5",
    ])
    assert args.idimage == "seed.jpg"
    assert args.outputpath == "output.png"


def test_invalid_sdxl_resolution_is_rejected():
    args = build_parser().parse_args(["--idimage", "a.jpg", "--outputpath", "b.png", "--width", "1025"])
    try:
        validate_arguments(args)
    except ValueError as error:
        assert "divisible by 8" in str(error)
    else:
        raise AssertionError("Expected invalid dimensions to fail")
