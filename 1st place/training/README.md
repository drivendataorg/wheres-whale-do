# Where-s-Whale-do_Competition (First Place Solution)
For this competition, your goal is to identify which images in a database contain the same individual beluga whale seen in a query image.

You will be provided with a set of queries, each one specifying a single query image of a beluga whale and a corresponding database to search for matches to that same individual. The database will include images of both matching and non-matching belugas. This is a learning-to-rank information retrieval task.

## PACKAGES
To make sure that the results are completely reproducable it is better to use the same versions and (Python 3.6.9):
- numpy == 1.19.5
- opencv == 4.5.4
- timm == 0.5.4
- albumentations == 1.1.0
- sklearn == 0.24.2
- torch == 1.7.1+cu110
- torchvision == 0.8.2+cu110
- pandas == 1.1.5

## Download weights
You can download the trained models from <a href="https://drive.google.com/drive/folders/1uHCh6ZOAmB6qULURklHse_CuoU-NpaQU?usp=sharing"> here</a>

## Training
### Using Training Scripts
* Clone the repo
* Add the data to the same directory inside the repo in a folder named data, which includes a sub folder called images and the metadata.csv
```
├── train_literal (source code)
├── train_top (source code)
└── data (dataset)
    └── images
        ├── img1.jpg
        .
        .
    ├── metadata.csv

```
  #### Train top models
  from inside the train_top folder run:
  ```
  from train import run
  run(0, "effb5")
  
  ```
  This will train fold 0 to train other folds for effb5 just change the number to 0,1,2,3,4.
  To train top model with effv2m run:
  ```
  from train import run
  run(0, "effv2")
  
  ```
  to train other folds just change the number to 0, 2, 4 (only three folds where trained (0, 2, 4))
  #### train literal models
  from inside the train_literal folder run:
  ```
  from train import run
  run(0, "effb5")
  
  ```
  This will train fold 0 to train other folds for effb5 just change the number to 0,1,2,4. 
  (only four folds are trained (0,1,2,4))
  #### PS: when after the training finishes change the name of weights files to avoid over writing it when starting a training for new fold.
### OR Using Notbooks:
* <a href="https://github.com/ammarali32/Where-s-Whale-do_Competition/blob/main/submissions/score_4661/training_notebooks/model_tf_efficientnet_b4_ns_IMG_SIZE_512_arcface_f0_7.16.ipynb"> Training effb5 models for top images </a>
* <a href="https://github.com/ammarali32/Where-s-Whale-do_Competition/blob/main/submissions/score_4661/training_notebooks/model_efficientnetv2_rw_m_IMG_SIZE_512_arcface_f2_6-79.ipynb"> Training effv2 models for top images </a>
* <a href="https://github.com/ammarali32/Where-s-Whale-do_Competition/blob/main/submissions/score_4903/training_nb/model_tf_efficientnet_b4_ns_IMG_SIZE_512_arcface_f4_literal_newapproach.ipynb"> Training effb5 models for literal </a>

There are some other notebooks which were used during experiments but none of them is used on the best or last submissions

## Inference
The final (eligable submission) could be found <a href="https://github.com/ammarali32/Where-s-Whale-do_Competition/blob/main/submissions/score_4887_final/main.py"> here</a> 
and another version of the same submission with a little refactoring is <a href="https://github.com/ammarali32/Where-s-Whale-do_Competition/blob/main/inference.py">here</a> but not tested.

 Other submissions could be found on the submissions folder.
## Specification

CPU intel corei9 10th generation

GPU RTX 3090

OS Linux

Memory 128 GB

Training Duration:
* Effb5 top about 180 minutes for a single fold (5 folds were trained)  in total about 15 hours
* Effv2 top about 220 minutes for a single folds (3 folds were trained) in total about 11 hours
* Effb5 literal about 180 minutes for a single fold (4 folds were trained) in total about 12 hours
Inference Duration:
It took almost 2 hours and 45 minutes on the driven data competition env
## Solution Architecture
![Alt Text](https://github.com/ammarali32/Where-s-Whale-do_Competition/blob/main/images/whale_diagram.JPG)
