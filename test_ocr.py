from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import requests


def load_env(path: Path = Path(".env")) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def parse_args() -> argparse.Namespace:
    load_env()
    ocr_base_url = os.getenv("RUNPOD_OCR_BASE_URL", "https://xgr9is1ahm59us-8501.proxy.runpod.net").rstrip("/")
    parser = argparse.ArgumentParser(description="Send 파일.png to the OCR API.")
    parser.add_argument(
        "--file",
        default="파일.png",
        help="Image file to OCR. Defaults to 파일.png in the current directory.",
    )
    parser.add_argument(
        "--url",
        default=f"{ocr_base_url}/ocr",
        help="OCR endpoint URL. Defaults to RUNPOD_OCR_BASE_URL/ocr.",
    )
    parser.add_argument("--timeout", type=int, default=600)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    image_path = Path(args.file)
    if not image_path.exists():
        raise SystemExit(f"File not found: {image_path.resolve()}")

    with image_path.open("rb") as image:
        response = requests.post(
            args.url,
            files={"file": (image_path.name, image, "image/png")},
            timeout=args.timeout,
        )

    print(f"status_code={response.status_code}")
    try:
        print(json.dumps(response.json(), ensure_ascii=False, indent=2))
    except ValueError:
        print(response.text)

    return 0 if response.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
