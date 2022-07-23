# Histological images for TMB index prediction
environment:
ubuntu18.04
python3.6

---
## Install dependencies
Run requirements.txt script to install all the dependencies.
```bash
# install openslide
sudo apt-get install openslide-tools
pip3 install openslide-python
# git clone software to ${SRC} and then run:
pip3 install -r ${SRC}/requirements.txt
```
## Prepare work

### 1、Preparation
```Bash
#/media/zw/sz_01/geneis_xml_update_by_Thorough_Images/colonrect_1 #WSI path
# annotated slide home
${DATASET}/lung/images
# class label home i.e., reg_tmb.csv
${DATASET}/lung/labels
# template image home i.e., template.png
${DATASET}/template
# get tile image home from raw slides
mkdir -p ${HOME}/tiles
mkdir -p ${HOME}/tiles_color_normalized
mkdir -p ${HOME}/model
```
A snapshot view of the directory structure is as follows (${DATASET}/lung/images):
<img src="./asset/folderview.png">

### 2、Prepare annotated svs slide dataset

Compute tile images, more details can be found in `config/config.py` you should specify ratio_train and ratio_test in your experiments. 

```bash
# dataset prepared
cd ${SRC}
# transmit parameter
sh ./HE_Auto.sh /tmp/data/lung/images /tmp/data/tiles /tmp/data/tiles_color_normalized /tmp/data/lung/labels/reg_tmb.csv ../asset/Template.png
```

---
## Training

Run the following command to start training more details can be found in  `config/config.py` as well as you should customize those parameters by yourself.

First of all, start a new tensorboard session for training investigation:

```Bash
tensorboard --logdir=runs --bind_all
```
After then, run train.py script and you can start training if everything is okay.
```Bash
python ${SRC}/train.py
```

Finally, model can be learned in ${DATASET}/model folder.

---
## Inference
Run test.py script for inference:
```Bash
python ${SRC}/test.py
```

