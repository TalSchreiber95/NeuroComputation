import staticAnalyzer
import os
from glob import glob
from pathlib import Path
import ujson as json
from tqdm import tqdm

from setting import config


def progress_bar(percent):
    for i in tqdm(range(100)):
        # Do some computation
        # ...
        # Update the progress bar with the percentage completed
        tqdm.update(percent - i)


def extractDataFromApkFiles():

    apkFolder = 'data/apks'

    input_path = config['input_path']
    images_dictionary = config['images_dictionary']
    


    dir_path = os.path.dirname(os.path.realpath(__file__))
    parent = Path(dir_path).parent
    path = '{parent}/{apkFolder}'.format(parent=parent, apkFolder=apkFolder)
    pathResult = '{path}/result'.format(path=path)

    if not os.path.exists(pathResult):
        os.makedirs(pathResult)
        print('create folder')

    jsonFile = '{pathResult}/data.json'.format(pathResult=pathResult)

    with open(jsonFile, 'w') as cleanFile:
        cleanFile.truncate(0)
        cleanFile.write(json.dumps([]))
        cleanFile.close()

    num_applications = len(glob(path + '/*.apk'))
    percent_increment = 100 / num_applications

    for r, d, f in os.walk(path):
        if num_applications == 0:
            break
        for file in f:
            label = 1
            if file.endswith(".png"):
                if file.endswith("_B.apk"):
                    label = 0

                filePath = os.path.join(r, file)
                print('Start working on file: ', file)
                print('Applications left:', num_applications)
                print('Progress: {:.2f}%'.format(
                    abs(100 - (percent_increment * num_applications))))
                num_applications -= 1

                try:
                    staticAnalyzer.run(filePath, path, '', label)
                    print()
                except Exception as e:
                    print(e)
                    continue


extractDataFromApkFiles()
