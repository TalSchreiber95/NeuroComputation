import numpy as np
from PIL import Image
import os
from setting import config
from utils import validateFileNames

images = ["bet1.png", "bet2.png", "bet3.png", "lamed1.png",
          "lamed2.png", "lamed3.png", "mem1.png", "mem2.png", "mem3.png"]
output_folder_path = 'data/output'
input_folder_path = 'data/input'


def image_to_vector(img, vector):
    width, height = img.size
    grid_size = (width // 10, height // 10)
    for j in range(10):
        for i in range(10):
            x = i * grid_size[0]
            y = j * grid_size[1]
            pixels = img.crop(
                (x, y, x + grid_size[0], y + grid_size[1])).getdata()
            if 255 in pixels:
                vector.append(1)
            else:
                vector.append(-1)
    return vector


def vector_to_image(vector):
    grid = np.reshape(vector, (10, 10))
    scaled_grid = np.kron(grid, np.ones((100, 100)))
    image = Image.fromarray(np.uint8(scaled_grid * 255))
    return image


def main():

    if os.path.exists(output_folder_path):
        file_list = os.listdir(output_folder_path)
        for file_name in file_list:
            file_path = os.path.join(output_folder_path, file_name)
            os.remove(file_path)

    input_path = config['input_path']
    images_dictionary = config['images_dictionary']

    for r, d, f in os.walk(input_path):
        for file in f:
            if validateFileNames(file, images_dictionary) == True:
                print('file:', file)
                
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

    for img in images:
        im = Image.open(f'{input_folder_path}/{img}')
        im_bw = im.convert("1")
        inverted_im_bw = im_bw.point(lambda x: 0 if x == 255 else 255)
        # inverted_im_bw.show()
        vector = []
        if img.startswith("bet"):
            filename = "bet"
            vector.append(1)
        elif img.startswith("lamed"):
            filename = "lamed"
            vector.append(2)
        elif img.startswith("mem"):
            filename = "mem"
            vector.append(3)

        returned_vector = image_to_vector(inverted_im_bw, vector)
        my_string = '(' + ', '.join(str(i) for i in returned_vector) + ')'
        with open(f'{output_folder_path}/{filename}', 'a+') as file:
            file.write(my_string + "\n")
        image = vector_to_image(returned_vector[1:])
        # image.show()

    for img in images:
        im = Image.open(f'{input_folder_path}/{img}')
        im_bw = im.convert("1")
        inverted_im_bw = im_bw.point(lambda x: 0 if x == 255 else 255)
        left_rotated_image = inverted_im_bw.rotate(15, expand=True)
        # left_rotated_image.show()
        vector = []
        if img.startswith("bet"):
            filename = "betLeft"
            vector.append(1)
        elif img.startswith("lamed"):
            filename = "lamedLeft"
            vector.append(2)
        elif img.startswith("mem"):
            filename = "memLeft"
            vector.append(3)
        returned_vector = image_to_vector(left_rotated_image, vector)
        my_string = '(' + ', '.join(str(i) for i in returned_vector) + ')'
        with open(f'{output_folder_path}/{filename}', 'a+') as file:
            file.write(my_string + "\n")
        image = vector_to_image(returned_vector[1:])
        image.show()

    for img in images:
        im = Image.open(f'{input_folder_path}/{img}')
        im_bw = im.convert("1")
        inverted_im_bw = im_bw.point(lambda x: 0 if x == 255 else 255)
        right_rotated_image = inverted_im_bw.rotate(-15, expand=True)
        # right_rotated_image.show()

        vector = []
        if img.startswith("bet"):
            filename = "betRight"
            vector.append(1)
        elif img.startswith("lamed"):
            filename = "lamedRight"
            vector.append(2)
        elif img.startswith("mem"):
            filename = "memRight"
            vector.append(3)
        returned_vector = image_to_vector(right_rotated_image, vector)
        my_string = '(' + ', '.join(str(i) for i in returned_vector) + ')'
        with open(f'{output_folder_path}/{filename}', 'a+') as file:
            file.write(my_string + "\n")
        image = vector_to_image(returned_vector[1:])
        # image.show()


if __name__ == "__main__":
    main()
