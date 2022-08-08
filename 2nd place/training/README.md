# Solution - Where's Whale-do?

Username: qwerty64

## Summary

Solution is based on sub-center ArcFace with Dynamic margins. As a final submission I used ensemble of 7 models which all use EfficientNet backbones. I trained models with different configurations to create diverse ensemble of models. At test time concatenated all 7 model embeddings and flipped embeddings before calculating cosine similarity. 

# Hardware

The solution was run on Google Colab using Tesla P100.

Training time: 10 hours (all models)

Inference time: 1 hour (all models)

# Run training

If necessary, modify data reading lines to specify the directory where the data is saved.
To start training run train.ipynb with uncommenting desired model configs.
The total model weights files that is saved out is 1,84 GB.

Trained model weights can be downloaded from this Google folder:
https://drive.google.com/file/d/19Bx8Uz8zEODXwdJs-nrCVhHl1g_so1uW/view?usp=sharing

# Run inference

After setting data directory same as competition format, with main.py and trained model checkpoints in same directory run main.py to generate submission.csv