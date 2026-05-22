from pathlib import Path
import shutil

if __name__ == '__main__':
    path_script = Path(__file__).parent
    files_list = list(path_script.rglob('03_*/*.ipynb'))
    new_files_list = [Path(str(file.resolve()).replace('CIC17', 'BCCC17')) for file in files_list]
    for fs, fd in zip(files_list, new_files_list):
        fd.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(fs, fd)