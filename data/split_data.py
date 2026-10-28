from pathlib import Path


DATA_DIR = Path(__file__).resolve().parent
SPLITS = ("train", "val", "test")
CLASSES = ("awake", "sleepy")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}


def main():
    missing = [
        DATA_DIR / split / class_name
        for split in SPLITS
        for class_name in CLASSES
        if not (DATA_DIR / split / class_name).is_dir()
    ]
    if missing:
        paths = "\n".join(f"  - {path}" for path in missing)
        raise SystemExit(f"Dataset folders are missing:\n{paths}")

    print(f"Using the existing pre-split dataset at: {DATA_DIR}")
    print("No files were copied or moved.")
    for split in SPLITS:
        counts = {
            class_name: sum(
                path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
                for path in (DATA_DIR / split / class_name).iterdir()
            )
            for class_name in CLASSES
        }
        print(f"{split}: awake={counts['awake']}, sleepy={counts['sleepy']}")


if __name__ == "__main__":
    main()