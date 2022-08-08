# Solution documentation
## Overview
* This is the solution docs for the `Where's Whale-do?` competition, if you have any question, please
contact Yang Xu (*****@gmail.com).
* Since the dataset is small, if you got different result while reproducing, please change another
seed and try retrain.

### Method
* CNN backbone + Arcface + Focal loss
* backbone: efficient b5, efficient b7 and convnext large

### Hardware
* Ryzen 3960x
* 128GB RAM
* 2xRTX 3090

### Software
* ubuntu server 20.04
* pytorch 1.9
* nvidia docker

## Reproduce
The file structure of the solution folder:
-input: input folder
--input/train_5folds_new.csv: fold split.
--input/individual_ids_new.json: whale_id to label map json.
--input/train_images_new: the training images (outputs of the preprocessing step)
--input/train_images: the original train images
-whale_chen: src folder
-results: output weights.

### 1. Training data
Please copy the training images to `input/train_images`.

### 2. Docker image build
a dockerfile had been added to fulfill the requirements. build the docker image first
```shell
cd docker
docker build -t whale:v0 .
cd ..
```

### 3. Train
to reproduce the training procedure, follow the steps below
run the docker container `whale:v0` with the mount of the input file
1. run docker
```shell
sudo docker run -v $PWD:/root --shm-size 8g -it --gpus all whale:v0 /bin/bash
# now we inside the container
cd /root
```
2. preprocess and resize the input image
```shell script
python prepreocessing.py
```
3. train the 3 models
```shell script
cd whale_chen
git config --global --add safe.directory /root/whale_chen
git checkout b5_sub
/bin/bash dist_hwd_train.sh
git checkout b7_sub
/bin/bash dist_hwd_train.sh
git checkout cl_sub
/bin/bash dist_hwd_train.sh
cd ..
```
The output weights is saved at results folder

### 4. Inference
I had already submit a zip file with code and weights, If you want to run this locally, Please follow the 
[official guide](https://github.com/drivendataorg/boem-belugas-runtime) of Code execution runtime. The submission file
is uploaded [here](https://drive.google.com/file/d/1JbEFWKBBGFgFHLVoq_lnzh03n9tsrlM-/view?usp=sharing), please ignore ipynb files at the `submission.zip`, weights in the `submission.zip` could be found at results dir. 

