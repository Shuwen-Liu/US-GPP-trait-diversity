"""Sort the files downloaded from the Zenodo archive into folders.

Zenodo stores files in a flat list, so folder names are encoded in the file
names with a double underscore: ``fd_layers_005__FRic_alpha.tif`` becomes
``fd_layers_005/FRic_alpha.tif``. Download every file of the archive into this
``data/`` folder, then run

    python organize_zenodo_files.py

from inside ``data/``. Files that are already in place are left alone.
"""
from pathlib import Path

here = Path(__file__).resolve().parent
moved = 0
for f in sorted(here.iterdir()):
    if not f.is_file() or "__" not in f.name:
        continue
    target = here.joinpath(*f.name.split("__"))
    target.parent.mkdir(parents=True, exist_ok=True)
    f.rename(target)
    moved += 1
    print(f"{f.name}  ->  {target.relative_to(here)}")
print(f"{moved} files organised")
