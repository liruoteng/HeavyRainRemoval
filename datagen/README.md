# Code for Heavy Rain Generation

## 0. Prerequisites
1. Matlab
2. Please download the background image datasets from [Dropbox Here](https://www.dropbox.com/sh/u0q5vjo2u2rf805/AABiiK5uoufbG7nGqF2cjc05a?dl=0) (409MB)
3. Extract the .gz file and put the two folders at `$project_root/datagen/RaindropDepth` and `$project_root/datagen/RaindropGroundtruth`

## 1. Generate Rain Datasets

1. run the following command in Matlab console: 
```
>> generate_dataset
```

2. It may takes a while to render rain streaks and rain veils on these datasets. After this process is done, you will see three new folders
in this directory

+ datagen
  + filelists
  + train
  + val
 
The `filelists` folder contains all the lists of image paths that will be used in the model training. The `train` folder contains all the training
images and groundtruths. `val` folder contains all images for validation purpose. 

3. Please adapt your actual data location in all the generated .txt files in the fileslits. 
 


