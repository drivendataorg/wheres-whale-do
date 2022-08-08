FROM pytorchlightning/pytorch_lightning:1.6.4-py3.9-torch1.10
RUN pip install -U albumentations==0.5.2 --no-deps
RUN pip install scikit-image
RUN pip install imgaug --no-deps
RUN pip install -U timm==0.5.4
RUN pip install opencv-python-headless
RUN pip install warmup_scheduler