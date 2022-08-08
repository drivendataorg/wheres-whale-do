# boem-beluga-whales

Install requirements in a Python 3 environment.

`pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html`

## Training 

Run `chmod +x src/*.sh`

Run `src/train.sh DATA_DIR MODEL_DIR`

e.g. `./src/train.sh /data/raw models`

## Inference

(optional) Download pretrained models (from Google Drive):

`gdown -O submission.zip 1-14Nvvj2LL6CYfl5xYo7X9AJCfx2IX1k`

`unzip -qqn submission.zip`

Run `python src/main.py`



