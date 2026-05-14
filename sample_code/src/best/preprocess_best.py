from pathlib import Path
import argparse
import subprocess
import sys

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


SRC_DIR = Path(__file__).resolve().parent.parent
ROOT_DIR = SRC_DIR.parent
CONFIG_PATH = ROOT_DIR / "best_models_config.toml"
PREPROC_DIR = SRC_DIR / "preprocessing"


def _load_config():
    with open(CONFIG_PATH, "rb") as f:
        return tomllib.load(f)


def _run(cmd: list[str]) -> None:
    print("\n[run]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _python_cmd(script_name: str, *args: str) -> list[str]:
    return [
        sys.executable,
        str(PREPROC_DIR / script_name),
        *args,
    ]


def preprocess_imu(cfg: dict, events: list[str]) -> None:
    _run(
        _python_cmd(
            "imu_preprocessing.py",
            "--run",
            "everything",
            "--events",
            *events,
            "--durations",
            str(cfg["window"]),
            "--overlaps",
            str(cfg["overlap"]),
            "--method",
            "FEAT",
            "--sensors",
            "Chest",
            "RightArm",
            "LeftArm",
            "--merge-sensors",
            "True",
        )
    )


def preprocess_infrared(cfg: dict, events: list[str]) -> None:
    _run(
        _python_cmd(
            "infrared_preprocessing.py",
            "--run",
            "everything",
            "--events",
            *events,
            "--durations",
            str(cfg["window"]),
            "--overlaps",
            str(cfg["overlap"]),
            "--method",
            "FEAT",
        )
    )


def preprocess_radar(cfg: dict, events: list[str]) -> None:
    _run(
        _python_cmd(
            "radar_preprocessing.py",
            "--run",
            "everything",
            "--events",
            *events,
            "--durations",
            str(cfg["window"]),
            "--overlaps",
            str(cfg["overlap"]),
            "--method",
            "FEAT",
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="run best preprocessing configs")
    parser.add_argument(
        "--data",
        type=str,
        nargs="+",
        default=["all"],
        choices=["all", "IMU", "INFRARED", "RADAR"],
        help="which data to preprocess",
    )
    parser.add_argument(
        "--events",
        type=str,
        nargs="+",
        default=["E03", "E04"],
        help="which emotion events to include",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = _load_config()

    selected = {"IMU", "INFRARED", "RADAR"} if "all" in args.data else set(args.data)

    if "IMU" in selected:
        preprocess_imu(config["imu"], args.events)
    if "INFRARED" in selected:
        preprocess_infrared(config["infrared"], args.events)
    if "RADAR" in selected:
        preprocess_radar(config["radar"], args.events)

    print("\n[done] best preprocessing artifacts prepared")


if __name__ == "__main__":
    main()
