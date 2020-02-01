#! /bin/python3


from pathlib import Path
import scipy.io as sio
import numpy as np
import cv2
import argparse

def prepare_darknet_label_and_image(image_path, polygons_per_image, output_dir, resized_dim):
    """Function that takes in an image path, polygons for that image, and a desired output directory.
    Loads the image to determine width and height information,
    iterates over every polygon (every hand), and determines the min/max X, Y coordinates,
    converts those min max XY values into relative values as needed by Darknet,
    determines a unique filename so that all the files can be placed together for easier processing,
    resizes the image and saves it.

    Args:
        image_path (pathlib.Path): Contains the path to the desired image to process.
        polygons_per_image (list of np.array): Contains multiple polygons, each polygon corresponds to
                                               one hand's worth of coordinates.
        output_dir (pathlib.Path): Contains the path to the desired output directory.
        resized_dim (tuple): Width, height. Determines the new size of the image. E.g. (608, 608)

    Returns:
        None
    """
    def generate_unique_filename(image_path):
        """Generates a unique filename from egohands dataset by concatenating the scnene name and filename.

        Args:
           image_path (pathlib.Path): Contains the path to the desired image to process.

        Returns:
            unique_filename (str): Unique filename, e.g. CARDS_COURTYARD_T_B_frame_1849
        """
        unique_filename = f"{image_path.parent.name}_{image_path.stem}"
        return unique_filename

    def find_minmax_xy(polygon_coords):
        # expects a 2D array, [[x1,y1],[x2,y2]..]
        x_min, y_min = np.amin(np.array(polygon_coords), axis=0)
        x_max, y_max = np.amax(np.array(polygon_coords), axis=0)
        return x_min, x_max, y_min, y_max

    def convert_minmax_to_darknet(x_min, x_max, y_min, y_max, img_width, img_height):
        """Convert min/max X Y values of the polygons to relative values of the img dimensions.
        Darknet expects <class> <x_center> <y_center> <width> <height>, where all values are relative.
        e.g. 0.716797 0.395833 0.216406 0.147222
        Reference: https://github.com/AlexeyAB/darknet#how-to-train-to-detect-your-custom-objects

        Args:
            x_min, x_max, y_min, y_max (int): Coordinates corresponding to the edges of the polygon.
            img_width, img_height (int): Image dimensions.

        Returns:
            (str): One line that represents a bounding box in darknet format, and the expected class.
                   For egohands, there is only 1 class, hence the class number is 0.
        """
        rec_x_center_rel = ((x_max - x_min)/2 + x_min)/img_width
        rec_y_center_rel = ((y_max - y_min)/2 + y_min)/img_height
        rec_width_rel = (x_max - x_min)/img_width
        rec_height_rel = (y_max - y_min)/img_height
        return f'0 {rec_x_center_rel} {rec_y_center_rel} {rec_width_rel} {rec_height_rel}'

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
            # some polygons are empty, not sure why, as I'm not familiar with .mat files.
            print(f'{image_path} has an empty polygon')
    label = '\n'.join(label)

    # determine output file location
    unique_filename = generate_unique_filename(image_path)
    label_path = Path(output_dir).joinpath(Path(unique_filename + '.txt'))
    output_image_path = Path(output_dir).joinpath(Path(unique_filename + '.jpg'))
    print(label_path, output_image_path)

    # resize image and save to new location and save label txt file to the same place
    img = cv2.imread(str(image_path))
    resized_img = cv2.resize(img, resized_dim)
    cv2.imwrite(str(output_image_path), resized_img)
    with open(label_path, 'w') as f:
        f.write(label)


def prepare_egohands(egohands_path, output_dir, resized_dim):
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
            prepare_darknet_label_and_image(image_path, list(all_polygons[i]), output_dir, resized_dim)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Egohands preparation script for Darknet.')
    parser.add_argument('egohands_path', type=str,
            help='Path to unzipped egohands directory.')
    parser.add_argument('--resized_dim', type=tuple, default=(608,608),
            help='(width, height) of resized image.')
    parser.add_argument('--output_dir', type=str, default='egohands_prepared',
            help="Path to store renamed egohands images and labels, defaults to 'egohands_prepared'.")
    args = parser.parse_args()
    prepare_egohands(args.egohands_path, args.output_dir, args.resized_dim)
