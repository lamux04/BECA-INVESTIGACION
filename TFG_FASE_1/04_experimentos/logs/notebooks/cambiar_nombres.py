from pathlib import Path
import shutil

if __name__ == '__main__':
    path_script = Path(__file__).parent
    files_list = list(path_script.rglob('*BCCC17*.sh'))
    for f in files_list:
        name_file = f.name
        shutil.copyfile(f, fd)