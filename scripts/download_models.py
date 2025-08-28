import os
from pathlib import Path
from typing import List
import requests
from tqdm import tqdm


BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"


class ModelSpec:
    def __init__(self, subdir: str, filename: str, urls: List[str]):
        self.subdir = subdir
        self.filename = filename
        self.urls = urls

    @property
    def dest_path(self) -> Path:
        return MODELS_DIR / self.subdir / self.filename


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def download_file(url: str, dest: Path, timeout: int = 60) -> bool:
    try:
        with requests.get(url, stream=True, timeout=timeout) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            tmp_path = dest.with_suffix(dest.suffix + ".part")
            with open(tmp_path, "wb") as f, tqdm(
                total=total if total > 0 else None,
                unit="B",
                unit_scale=True,
                desc=f"{dest.name}"
            ) as pbar:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        if total:
                            pbar.update(len(chunk))
            tmp_path.replace(dest)
        return True
    except Exception as e:
        print(f"  Failed: {url} -> {e}")
        return False


def download_with_fallbacks(spec: ModelSpec) -> None:
    ensure_dir(spec.dest_path.parent)
    if spec.dest_path.exists() and spec.dest_path.stat().st_size > 0:
        print(f"Exists: {spec.dest_path}")
        return
    print(f"Downloading: {spec.dest_path}")
    for url in spec.urls:
        print(f"  Trying: {url}")
        if download_file(url, spec.dest_path):
            print(f"  Saved: {spec.dest_path}")
            return
    print(f"  Skipped: Could not download {spec.filename}. See README for manual links.")


def main():
    specs: List[ModelSpec] = [
        # OpenCV DNN face detector (reliable official sources)
        ModelSpec(
            subdir="face",
            filename="deploy.prototxt",
            urls=[
                "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
                "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/dnn/face_detector/deploy.prototxt",
            ],
        ),
        ModelSpec(
            subdir="face",
            filename="res10_300x300_ssd_iter_140000.caffemodel",
            urls=[
                "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel",
            ],
        ),

        # Age/Gender Caffe models (community mirrors; may require manual download if mirrors change)
        ModelSpec(
            subdir="age",
            filename="age_deploy.prototxt",
            urls=[
                "https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/age_deploy.prototxt",
                "https://raw.githubusercontent.com/collabH/age-gender/master/age/age_deploy.prototxt",
            ],
        ),
        ModelSpec(
            subdir="age",
            filename="age_net.caffemodel",
            urls=[
                # LearnOpenCV (historical)
                "https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/age_net.caffemodel",
                # OpenCV 3rdparty (possible locations depending on branch history)
                "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_age_gender_20170830/age_net.caffemodel",
                "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_age_gender/age_net.caffemodel",
                # Community mirror
                "https://github.com/PRITHASAMANTA/Age-gender-detection/raw/main/age_net.caffemodel",
            ],
        ),
        ModelSpec(
            subdir="gender",
            filename="gender_deploy.prototxt",
            urls=[
                "https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/gender_deploy.prototxt",
                "https://raw.githubusercontent.com/collabH/age-gender/master/gender/gender_deploy.prototxt",
            ],
        ),
        ModelSpec(
            subdir="gender",
            filename="gender_net.caffemodel",
            urls=[
                # LearnOpenCV (historical)
                "https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/gender_net.caffemodel",
                # OpenCV 3rdparty (possible locations depending on branch history)
                "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_age_gender_20170830/gender_net.caffemodel",
                "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_age_gender/gender_net.caffemodel",
                # Community mirror
                "https://github.com/PRITHASAMANTA/Age-gender-detection/raw/main/gender_net.caffemodel",
            ],
        ),
    ]

    for spec in specs:
        download_with_fallbacks(spec)

    print("\nDone. If some models failed to download, see README for manual download links.")


if __name__ == "__main__":
    main()


