from pathlib import Path
import tarfile


model_root = Path(__file__).resolve().parents[1]
archive_path = model_root / "pilot_source.tar.gz"
source_directories = (
    "callbacks",
    "configs",
    "e3moldiffusion",
    "experiments",
    "kinodata",
    "posebusters",
    "posecheck",
    "scripts",
)

with tarfile.open(archive_path, "w:gz") as archive:
    for directory in source_directories:
        archive.add(model_root / directory, arcname=directory)

print("PILOT/pilot_source.tar.gz")
