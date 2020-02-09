# What is this repository?
It contains helper scripts for using the egohands dataset on Darknet for hand recognition training.  
  
  
|Script|Purpose|Arguments|
|---|---|---|
|prepare_egohands.py|Converts the polygons of the egohands dataset into the 'object-class x_center y_center width height' that's needed by darknet. E.g. 0 0.3215 0.8218 0.072 0.1006| path to egohands directory, resolution and output directory|
|generate_train_test.py|Darknet requires a train.txt and test.txt each containing lines of relative file paths pointing to the image that is part of train/test. This script generates these 2 files.|Relative path from darknet executable to prepared egohands images, seed and percentage of training data.|
  


# How to use
1. Download the egohands dataset (labelled data): http://vision.soic.indiana.edu/projects/egohands/
1. Unzip it, you should have a directory called egohands with _LABELLED_SAMPLES inside.
1. Create a virtenv and install scipy and cv2.
1. ```python prepare_egohands.py ./egohands # Optional params: Different resolutions, Output directory.```
1. A directory called egohands_prepared should be created, and you can monitor the boxes with the pop up window.
1. ```python generate_train_test.py ../egohands_darknet/egohands_prepared # seed, training percentage```
1. Now you have resized images as well as the darknet format labels in one directory, and train.txt and test.txt to facilitate darknet training.
