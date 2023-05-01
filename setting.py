
import os
import platform
from glob import glob
from pathlib import Path
import ujson as json
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


_project_path = os.path.dirname(os.path.realpath(__file__))
input_path = f'{_project_path}data/input'
output_path = f'{_project_path}data/output'
images_dictionary = ["bet", "lamed", "mem"]

config = {
    'input_path': input_path,
    'output_path': output_path,
    'images_dictionary': images_dictionary,
}
