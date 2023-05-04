import numpy as np
from PIL import Image
import os
from setting import config
from utils import validateFileNames


output_path = f'{config["output_path"]}'
input_path = f'{config["input_path"]}/images'



import os


def main():
    rotate_path = config['rotate_path']
    normal_path = config['normal_path']
    output_result_path = config['output_result_path']
    result = ''
    with open(output_result_path, 'w') as output_file:
        for r, d, f in os.walk(rotate_path):
            # Loop through each file path
            for file_path in f:
                    # Open file in read mode
                with open(f'{rotate_path}/{file_path}', 'r') as input_file:
                    # Loop through each line in the file and write to output file
                    for line in input_file:
                        if line.strip() and line.startswith('(') :
                            result += line

        for r, d, f in os.walk(normal_path):
            # Loop through each file path
            for file_path in f:
                # Loop through each line in the file and write to output file
                with open(f'{normal_path}/{file_path}', 'r') as input_file:
                    for line in input_file:
                        if line.strip() and line.startswith('(') :
                            result += line

        output_file.write(result.replace(" ","").replace(")(", ")\n("))
    output_file.close()
    print('All files have been combined into ' + output_result_path)
                
   
            


if __name__ == "__main__":
    main()
