import os
import platform
from glob import glob
from pathlib import Path
import ujson as json
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

apkFolder = 'apks'
jarFolder = 'jarFiles'
trainFolder = 'train'
testFolder = 'test'


_project_path = os.path.dirname(os.path.realpath(__file__))
jarPath = f'{_project_path}/{jarFolder}'
apksPath = f'{_project_path}/data/{apkFolder}'
resultApksPath = f'{_project_path}/data/{apkFolder}/result'
apksResultJsonPath = f'{_project_path}/data/{apkFolder}/result/data.json'
trainPath = f'{_project_path}/data/{trainFolder}'
testPath = f'{_project_path}/data/{testFolder}'
featureExtractorPath = f'{_project_path}/featureExtractor'
AAPT = f"{_project_path}/{featureExtractorPath}/aapt"
APICALLS = "APIcalls.txt"
BACKSMALI = "baksmali-2.0.3.jar"  # location of the baksmali.jar file
ADSLIBS = "ads.csv"


config = {
    'AAPT': AAPT,
    'APICALLS': APICALLS,
    'BACKSMALI': BACKSMALI,
    'ADSLIBS': ADSLIBS,
    '_project_path': _project_path,
    'jarPath': jarPath,
    'apksPath': apksPath,
    'trainPath': trainPath,
    'testPath': testPath,
    'featureExtractorPath': featureExtractorPath,
    'apksResultJsonPath': apksResultJsonPath,
    'resultApksPath': resultApksPath
}