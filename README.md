# Body morphometry for sarcopenia (Kaggle Competition)
## MOAI 2020 Body Morphometry AI Segmentation Online Challenge
Private Leaderboard : 2nd (Hallym MMC)
[Competition Page] (https://www.kaggle.com/c/body-morphometry-for-sarcopenia/overview)


# body-morp-segmentation

Please see the attached pdf file named '파이프라인 설명서.pdf'

## Model Zoo
- AlbuNet (resnet34) from [\[ternausnets\]](https://github.com/ternaus/TernausNet)
- Resnet50 from [\[selim_sef SpaceNet 4\]](https://github.com/SpaceNetChallenge/SpaceNet_Off_Nadir_Solutions/tree/master/selim_sef/zoo)
- SCSEUnet (seresnext50) from [\[selim_sef SpaceNet 4\]](https://github.com/SpaceNetChallenge/SpaceNet_Off_Nadir_Solutions/tree/master/selim_sef/zoo)


## Original source code

https://github.com/sneddy/pneumothorax-segmentation

Video with short explanation: https://youtu.be/Wuf0wE3Mrxg

Presentation with short explanation: https://yadi.sk/i/oDYnpvMhqi8a7w

Competition: https://kaggle.com/c/siim-acr-pneumothorax-segmentation


## Main Features

### Combo loss
Used \[[combo loss\]](https://github.com/SpaceNetChallenge/SpaceNet_Off_Nadir_Solutions/blob/master/selim_sef/training/losses.py) combinations of BCE, dice and focal. In the best experiments the weights of (BCE, dice, focal), that I used were:
- (3,1,4) for albunet_valid and seunet;
- (1,1,1) for albunet_public;
- (2,1,2) for resnet50.

**Why exactly these weights?**

In the beginning, I trained using only 1-1-1 scheme and this way I get my best public score.

I noticed that in older epochs, Dice loss is higher than the rest about 10 times.

For balancing them I decide to use a 3-1-4 scheme and it got me the best validation score.

As a compromise I chose 2-1-2 scheme for resnet50)

### Checkpoints averaging
Top3 checkpoints averaging from each fold from each pipeline on inference

### Horizontal flip TTA

## File structure
    ├── unet_pipeline
    │   ├── experiments
    │   │   ├── some_experiment
    │   │   │   ├── train_config.yaml
    │   │   │   ├── inference_config.yaml
    │   │   │   ├── submit_config.yaml
    │   │   │   ├── checkpoints
    │   │   │   │   ├── fold_i
    │   │   │   │   │   ├──topk_checkpoint_from_fold_i_epoch_k.pth 
    │   │   │   │   │   ├──summary.csv
    │   │   │   │   ├──best_checkpoint_from_fold_i.pth
    │   │   │   ├── log
    ├── input                
    │   ├── dicom_train
    │   │   ├── some_folder
    │   │   │   ├── some_folder
    │   │   │   │   ├── some_train_file.dcm
    │   ├── dicom_test   
    │   │   ├── some_folder
    │   │   │   ├── some_folder
    │   │   │   │   ├── some_test_file.dcm
    |   ├── new_sample_submission.csv
    │   └── new_train_rle.csv
    └── requirements.txt

## Install
```bash
pip install -r requirements.txt
```

## Data Preparation

```bash
kaggle competitions download -c body-~~~
```

## Pipeline launch example
Training:
```bash
cd unet_pipeline
python Train.py experiments/albunet512/train_config_part0.yaml
python Train.py experiments/albunet512/train_config_part1.yaml
```
As an output, we get a checkpoints in corresponding folder.


Inference:
```bash
cd unet_pipeline
python Inference.py experiments/albunet512/inference_config.yaml
```
As an output, we get a pickle-file with mapping the file name into a mask with pneumothorax probabilities.

Submit:
```bash
cd unet_pipeline
python TripletSubmit.py experiments/albunet512/submit.yaml
```
As an output, we get submission file with rle.
