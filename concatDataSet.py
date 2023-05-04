import numpy as np
from PIL import Image
import os
from setting import config
from utils import validateFileNames


output_path = f'{config["output_path"]}'
input_path = f'{config["input_path"]}/images'




def main():


    images_dictionary = config['images_dictionary']
    for r, d, f in os.walk(input_path):

        for file in f:
            if validateFileNames(file, images_dictionary) == True:
                filePathInput = f'{input_path}/{file}'
                filePathOutput = f'{output_path}'
                name = file
                im = Image.open(filePathInput)
                im_bw = im.convert("1")
                inverted_im_bw = im_bw.point(lambda x: 0 if x == 255 else 255)
                # inverted_im_bw.show()
                vector = []
                if name.startswith("bet"):
                    filename = "bet"
                    vector.append(1)
                elif name.startswith("lamed"):
                    filename = "lamed"
                    vector.append(2)
                elif name.startswith("mem"):
                    filename = "mem"
                    vector.append(3)

                returned_vector = image_to_vector(inverted_im_bw, vector)
                my_string = '(' + ', '.join(str(i) for i in returned_vector) + ')'
                with open(f'{filePathOutput}/{filename}.txt', 'a+') as file:
                    file.write(my_string + "\n")
                with open(f'{filePathOutput}/result.txt', 'a+') as file:
                    file.write(my_string + "\n")
                image = vector_to_image(returned_vector[1:])
                # image.show()

                im = Image.open(filePathInput)
                im_bw = im.convert("1")
                inverted_im_bw = im_bw.point(lambda x: 0 if x == 255 else 255)
                left_rotated_image = inverted_im_bw.rotate(15, expand=True)
                # left_rotated_image.show()
                vector = []
                if name.startswith("bet"):
                    filename = "betLeft"
                    vector.append(1)
                elif name.startswith("lamed"):
                    filename = "lamedLeft"
                    vector.append(2)
                elif name.startswith("mem"):
                    filename = "memLeft"
                    vector.append(3)
                returned_vector = image_to_vector(left_rotated_image, vector)
                my_string = '(' + ', '.join(str(i) for i in returned_vector) + ')'
                with open(f'{filePathOutput}/{filename}.txt', 'a+') as file:
                    file.write(my_string + "\n")
                with open(f'{filePathOutput}/result.txt', 'a+') as file:
                    file.write(my_string + "\n")
                image = vector_to_image(returned_vector[1:])
                # image.show()

            
                im = Image.open(filePathInput)
                im_bw = im.convert("1")
                inverted_im_bw = im_bw.point(lambda x: 0 if x == 255 else 255)
                right_rotated_image = inverted_im_bw.rotate(-15, expand=True)
                # right_rotated_image.show()

                vector = []
                if name.startswith("bet"):
                    filename = "betRight"
                    vector.append(1)
                elif name.startswith("lamed"):
                    filename = "lamedRight"
                    vector.append(2)
                elif name.startswith("mem"):
                    filename = "memRight"
                    vector.append(3)
                returned_vector = image_to_vector(right_rotated_image, vector)
                my_string = '(' + ', '.join(str(i) for i in returned_vector) + ')'
                with open(f'{filePathOutput}/{filename}.txt', 'a+') as file:
                    file.write(my_string + "\n")
                with open(f'{filePathOutput}/result.txt', 'a+') as file:
                    file.write(my_string + "\n")
                image = vector_to_image(returned_vector[1:])
                # image.show()
            else: 
                print(f'File name is invalid {file}')


if __name__ == "__main__":
    main()
