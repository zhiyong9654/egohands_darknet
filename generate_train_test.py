#! /bin/python3
import random
from pathlib import Path
import argparse

def generate_train_test(rel_path_to_img_from_darknet, path_to_processed, seed, train_ratio=0.7):
    """ Function to generate 2 txt files, each line containing the relative
        path to an image from darknet. E.g. data/obj/img1.jpg.

        Args:
            rel_path_to_img_from_darknet (str): Relative path to processed
                images from darknet exe. E.g. '../code_base/egohands_prepared/'
            path_to_processed (str): Path to directories containing images and labels.
            seed (int/False): If seed is defined, it will be used to ensure data is
                split the same way.

        Returns:
            None
    """
    if seed:
        random.seed(seed)
    img_paths = list(Path(path_to_processed).glob('*.jpg')) 
    img_paths = [str(Path(rel_path_to_img_from_darknet).joinpath(path.name)) for path in img_paths]
    random.shuffle(img_paths)
    train_paths = img_paths[:int(train_ratio*len(img_paths))]
    test_paths = img_paths[int(train_ratio*len(img_paths)):]
    with open('train.txt', 'w') as f:
        f.write('\n'.join(train_paths))
    with open('test.txt', 'w') as f:
        f.write('\n'.join(test_paths))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Egohands preparation script for Darknet.')
    parser.add_argument('rel_path', type=str,
            help='Relative path to egohands_prepared from darknet executable.')
    parser.add_argument('path_to_processed', type=str,
            help='Path to egohands_prepared.')
    parser.add_argument('--seed', type=int, default=False,
            help='Set seed to a desired integer if you want train test splits to be repeatable')
    parser.add_argument('--train_ratio', type=float, default=0.7,
            help='Set percentage of training data here. Defaults to 0.7.')
    args = parser.parse_args()
    generate_train_test(args.rel_path, args.path_to_processed, args.seed, args.train_ratio)
