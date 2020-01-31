#! /bin/python3

from pathlib import Path
import shutil
import scipy.io as sio
import numpy as np
import cv2
import argparse

def prepare_darknet_label_and_image(image_path, polygons_per_image, output_dir):
    # darknet expects a txt label file with relative coords of bounding boxes.
    # It also expects the img file to share the same name as the txt file.
    def generate_unique_filename(image_path):
        unique_filename = f"{image_path.parent.name}_{image_path.stem}"
        return unique_filename

    def find_minmax_xy(polygon_coords):
        # expects a 2D array, [[x1,y1],[x2,y2]..]
        x_min, y_min = np.amin(np.array(polygon_coords), axis=0)
        x_max, y_max = np.amax(np.array(polygon_coords), axis=0)
        return x_min, x_max, y_min, y_max

    def convert_minmax_to_darknet(x_min, x_max, y_min, y_max, img_width, img_height):
        # Darknet expects <class> <x_center> <y_center> <width> <height>, where all values are relative.
        # e.g. 0.716797 0.395833 0.216406 0.147222
        # Reference: https://github.com/AlexeyAB/darknet#how-to-train-to-detect-your-custom-objects
        rec_x_center_rel = ((x_max - x_min)/2 + x_min)/img_width
        rec_y_center_rel = ((y_max - y_min)/2 + y_min)/img_height
        rec_width_rel = (x_max - x_min)/img_width
        rec_height_rel = (y_max - y_min)/img_height
        # There's only one class in the egohands dataset, therefore class is 0.
        return f'0 {rec_x_center_rel} {rec_y_center_rel} {rec_width_rel} {rec_height_rel}'

    
    # load image to get height and width data. CV2 can only load a string.
    img = cv2.imread(str(image_path))
    # generate bounding boxes info - labels in darknet format, note that an image can have multiple polygons (multiple hands)
    label = []
    for polygon in polygons_per_image:
        try:
            x_min, x_max, y_min, y_max = find_minmax_xy(polygon)
            label.append(convert_minmax_to_darknet(x_min, x_max, y_min, y_max, np.size(img, 1), np.size(img, 0)))
            # verify rectangle
            image_to_verify = cv2.rectangle(img, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (255, 0, 0), 2)
            cv2.imshow('verification_window', image_to_verify)
            cv2.waitKey(2)
        except ValueError:
            print(f'{image_path} has an empty polygon')
    label = '\n'.join(label)
    
    # determine output file location
    unique_filename = generate_unique_filename(image_path)
    label_path = Path(output_dir).joinpath(Path(unique_filename + '.txt'))
    output_image_path = Path(output_dir).joinpath(Path(unique_filename + '.jpg'))
    print(label_path, output_image_path)

    # cp image to new location and save label txt file to the same place
    shutil.copy(image_path, output_image_path)
    with open(label_path, 'w') as f:
        f.write(label)
    
    
def prepare_egohands(egohands_path, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir()
    p = Path(egohands_path).joinpath('_LABELLED_SAMPLES')
    for scene_path in p.iterdir():
        # sort images as .m file stores points in alphabetical order
        images_per_scene = list(scene_path.glob('*.jpg'))
        images_per_scene.sort()
        # load all polygons for every image
        all_polygons = sio.loadmat(next(scene_path.glob('*.mat')))['polygons'][0]
        for i, image_path in enumerate(images_per_scene):
            prepare_darknet_label_and_image(image_path, list(all_polygons[i]), output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Egohands preparation script for Darknet.')
    parser.add_argument('egohands_path', type=str, 
            help='Path to unzipped egohands directory.')
    parser.add_argument('--output_dir', type=str, default='egohands_prepared', 
            help="Path to store renamed egohands images and labels, defaults to 'egohands_prepared'.")
    args = parser.parse_args()
    prepare_egohands(args.egohands_path, args.output_dir)
