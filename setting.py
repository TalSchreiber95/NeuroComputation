
import os
import platform
from glob import glob
from pathlib import Path
import ujson as json
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


_project_path = os.path.dirname(os.path.realpath(__file__))
input_path = f'{_project_path}/dataSet/input'
output_path = f'{_project_path}/dataSet/output'
output_result_path = f'{_project_path}/dataSet/output/result.txt'
normal_path = f'{input_path}/normal'
rotate_path = f'{input_path}/rotate'

images_dictionary = ["bet", "lamed", "mem"]
config = {
    'input_path': input_path,
    'output_path': output_path,
    'images_dictionary': images_dictionary,
    'output_result_path': output_result_path,
    'normal_path': normal_path,
    'rotate_path': rotate_path,
}
