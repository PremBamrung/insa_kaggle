from glob import glob
import zipfile
import os


def unzip():
    zip_files = glob('*.zip')
    zip_files
    # unzip
    for file in zip_files:
        with zipfile.ZipFile(file, 'r') as zip_ref:
            zip_ref.extractall(os.getcwd())
    # delete zip file
    for file in zip_files:
        os.remove(file)


unzip()
