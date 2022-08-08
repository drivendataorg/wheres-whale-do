# Winners of the Where's Whale-do? Competition
[<img src='https://s3.amazonaws.com/drivendata-public-assets/logo-white-blue.png' width='600'>](https://www.drivendata.org/)
![Banner Image](https://drivendata-public-assets.s3.amazonaws.com/boem-beluga-pod.jpeg)

## Goal of the Competition
**The goal of [this challenge](https://www.drivendata.org/competitions/96/beluga-whales/page/478/) is to help wildlife researchers accurately identify endangered Cook Inlet beluga whale individuals from photographic images.** Specifically, the task is [query ranking](https://en.wikipedia.org/wiki/Ranking_(information_retrieval)) for images of individual whales given a query image against an image database, which is a key step in the full photo-identification process.

This was a code execution challenge, which means that participants packaged the files needed to perform inference and submitted them for containerized execution.  

### Background

Beluga whales are sociable mammals that live in pods and are known for being one of the [most vocal of all whales](https://www.worldwildlife.org/species/beluga). Measuring up to 15 feet and 3,500 pounds, they are found in seasonally ice-covered waters throughout the arctic and sub-arctic regions of the northern hemisphere. Belugas are legally protected in the United States and the [National Oceanic and Atmospheric Administration (NOAA) Fisheries](https://www.fisheries.noaa.gov/) monitors [five different populations of belugas](https://www.fisheries.noaa.gov/species/beluga-whale) across Alaskan waters, with a focus on the Cook Inlet belugas.

Cook Inlet beluga whales were listed as an endangered species under the Endangered Species Act in 2008 and are at risk for extinction. The Marine Mammal Laboratory at the NOAA Alaska Fishery Science Center began conducting an annual [photo-identification](https://en.wikipedia.org/wiki/Wildlife_photo-identification) survey of Cook Inlet belugas to more closely monitor and track individual whales. The Lab takes overhead photographs of these belugas using drones and lateral photographs using vessels.

Processing and analyzing new survey images of Cook Inlet belugas is largely manual and consumes significant time and resources. New and improved methods are needed to help automate this process and accurately find matches of the same individual whale in different survey images.


## What's in this Repository

This repository contains `training` and `inference` code from winning competitors in the [Where's Whale-do?](https://www.drivendata.org/competitions/96/beluga-whales/page/478/) challenge. Inference was performed in the execution runtime, which is specified [runtime repository](https://github.com/drivendataorg/boem-belugas-runtime). Training of models is generally done using participant's own resources.

Additional solution documentation can be found in the `reports` folder inside the directory for each submission.


## Winning Submissions

Place |Team or User | Public Score | Private Score | Summary of Model | Bonus Round
--- | --- |--------------|---------------| --- | ---
1   | Ammarali32    | 0.4902       | 0.4954        | Ensemble of pre-trained EfficientNet backbones, trained with k-fold cross validation and ArcFace loss. Matching database images are used for re-ranking, and horizontal flip augmentation is applied during inference. | [Grad-CAM heatmaps](https://github.com/drivendataorg/wheres-whale-do/tree/main/Explainability%20Bonus/1st%20place)
2   | qwerty64 | 0.4936       | 0.4953        | Ensemble of pre-trained EfficientNet backbones, trained with k-fold cross validation and sub-center ArcFace with adaptive margin loss.
3   | sheep | 0.4846       | 0.4910        | Ensemble of pre-trained ConvNext and EfficientNet backbones, trained with k-fold cross validation and Focal Loss.
4   | karelds | 0.4838     | 0.4871        | Ensemble of pre-trained EfficientNet backbones, trained with k-fold cross validation and sub-center ArcFace with adaptive margin loss. Horizontal flip augmentation during inference. | [Grad-CAM heatmaps](https://github.com/drivendataorg/wheres-whale-do/tree/main/Explainability%20Bonus/4th%20place)


### Additional Information

* [Where's Whale-do Challenge](https://www.drivendata.org/competitions/96/beluga-whales/page/478/)
* [Benchmark Blog Post](https://drivendata.co/blog/belugas-benchmark)
* [Winners Blog Post](TBD)

Winning code for other DrivenData competitions is available in the [competition-winners repository](https://github.com/drivendataorg/competition-winners).