
MMDL 
===========
### Predicting colorectal cancer tumor mutational burden from histopathological images and clinical information using multi-modal deep learning 
*Bioinformatics*

* This study proposed a multi-modal deep learning model based on residual network (resnet) and multi-modal compact bilinear pooling (MCB), to predict TMB status (i.e., TMB_H vs TMB_L) directly from histopathological images and clinical data. We applied the model in CRC data from the cancer genome atlas (TCGA), and compared it with other four popular methods including resnet18, resnet50, vgg19 and alexnet. We also tried different TMB threshold at a few percentiles (10%, 14.3%, 15%, 16.3%, 20%, 30% and 50%) to differentiate TMB_H and TMB_L. For percentile 14.3% (i.e., TMB value 20), our model achieved an area under the receiver operating characteristic (ROC) curve (AUC) of 0.817 in the 5-fold cross-validation, which is better than that of other compared models. In addition, we also found that TMB values are significantly associated with tumor stage, and N and M staging. Our study shows that deep learning models can predict TMB status from histopathological images and clinical information only, which is worth clinical application.*

© This code is made available for non-commercial academic purposes. 

## MMDL: Multi-modal Deep learning

## Pre-requisites:
* Linux (Tested on Ubuntu 22.04)
* Anaconda3 
* Python (3.8.13), Python (3.8.13), torch(1.7.1), torchaudio(0.7.2), torchvision(0.8.2), openslide-python(1.1.2), matplotlib(3.5.2), opencv-python(4.6.0.66), pandas(1.4.3), scikit-learn(0.23.2), tiatoolbox(0.4.0), tensorboard(2.9.1).

### Installation Guide for Linux (using anaconda)
* sudo apt-get install python3-dev
* sudo apt-get install gcc
* sudo apt install openslide-tools
* sudo apt install python3-openslide
* python -m pip install --upgrade pip
* pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
* pip install torch==1.7.1
* pip install torchaudio==0.7.2
* pip install torchvision==0.8.2
* pip install openslide-python==1.1.2
* pip install matplotlib
* pip install opencv-python
* pip install pandas
* pip install scikit-learn
* pip install tiatoolbox==0.4.0
* pip install tensorboard
* pip install seaborn
* [Installation Guide](https://github.com/hkmgeneis/MMDL/IMMDL.txt)

### Data Preparation
We first saved hematoxylin-eosin(H&E) stained histopathological whole-section images(WSI) in svs format. Professional pathologists used ASAP software to annotate tumor areas and save the annotation information in .xml files. Here, it is ensured that the WSI and annotation information of the same patient were placed in the same folder (named after the patient id(e.g.TCGA-3L-AA1B)).The following folder structure is assumed:
```bash
slide_image_root/
    └──WSICRCTMB/
         ├── patient_id1
                ├── WSI_1.svs
                ├── WSI1_annotated_information.xml
                └── ...
         └── patient_id2
                ├── WSI_1.svs
                ├── WSI1_annotated_information.xml
                └── ...
         └── ...
```
slide_image_root is the base directory of all datasets. WSICRCTMB is the name of the folder containing data specific to one experiment from each patient is stored as .svs and .xml files.

### Segementation
``` shell
python slide_to_tiles.py 
```
The following arguments need to be specified:
* slide_image_root (str): Path to the patients' WSI data.
* tiles_image_root (str): Path to saved .png tiles for the WSI data.
* size_square (int): The size of tiles, the default is "512".
* prepare_types (str): WSI type, default ".svs".

### Color Normalization
``` shell
python stain_color_norm.py 
```
The following arguments need to be specified:
* source_input (str): Path to saved .png tiles for the WSI data, same as "tiles_image_root" in slide_to_tiles.py.
* target_input (list of str): Standard color-normalized sample tiles.
* output_dir (str): Path to saved .png tiles for color normalization.
* file_types (str): Tiles type, default "*.png".

### label
Datasets are expected to be prepared in two csv formats. One of the csv file(e.g.colo_tmb_label4.csv) contains at least 3 columns: **TCGA_ID**, **TMB_Value**, **TMB20**. Each **TCGA_ID** is a unique identifier for a patient, which corresponds to a **TMB_Value**. **TMB20** refers to dividing TMB high and low states with a TMB value of 20 as a threshold. Those with a TMB value greater than or equal to 20 are classified as high TMB and marked as 1. Those with a TMB value less than 20 are classified as low TMB and marked as 0. We provide a dummy example of a data csv file in the **label** folder, named **crc_tmb_label4.csv**. Another csv file is the clinical information table corresponding to the patient. Each **TCGA_ID** is a unique identifier for a patient. First, the clinical information of the patients was obtained and matched with the patients in this study. Simultaneous data cleaning, including dropping columns and filling missing values. Delete some clinical feature columns that are not meaningful to this study (such as bar-code, uuid, etc.), blank, not report or not available more than 25%. The clinical information table was then converted into a numerical list using one-hot encoding. For continuous features, fill missing values with the mean. Discontinuous features, filled with mode. We provide a dummy example of a dataset csv file in the **label** folder, named **clinic4_35.csv**. 

For H&E training, look under main.py:
```python
def main(ocs, classification, K, cnv):
    for k in range(K):
        generative_model("resnet18", k, cnv=cnv)  
        #path = os.getcwd()+f'/results/resnet18_{k}' 
        #if cnv:
            #path = path + '_cnv'
        #model_ft = torch.load(path + '.pkl')
        #test(model_ft, "resnet18", k)
```
In addition to the fold of cross-validation (k), the following arguments need to be specified:
* path (str): Path to save H&E image training models.
* cnv (bool): "False" indicates that only H&E image features are used to predict tumor TMB, "True" means fusion of H&E images and clinical features to predict tumor TMB.

### Training Splits
For evaluating the algorithm's performance, we randomly partitioned our dataset into training and test splits using cross-validation. An example splits for the dummy data can be fould in **data**. These splits can be automatically generated using the train_test_splitter.py script with minimal modification. For example, the dummy splits were created by calling:
 
``` shell
python train_test_splitter.py 
```
The following arguments need to be specified:
* stained_tiles_home (str): Path to saved .png tiles for color normalization, same as "toutput_dir" in stain_color_norm.py.
* label_dir_path (str): Path to the label csv file. We provide a dummy example of a data csv file in the **label** folder, named **crc_tmb_label4.csv**. 

### Training
``` shell
python main.py 
```
Other arguments such as --momentum, --step_size, --lr, --gamma, --dataset_sizes, --num_epochs and --available_policies can be specified to customize your experiments. By default results will be saved to **results**.

### Test
For H&E test, look under main.py:
```python
def main(ocs, classification, K, cnv):
    for k in range(K):
        #generative_model("resnet18", k, cnv=cnv)  
        path = os.getcwd()+f'/results/resnet18_{k}' 
        if cnv:
            path = path + '_cnv'
        model_ft = torch.load(path + '.pkl')
        test(model_ft, "resnet18", k)
```
In addition to the fold of cross-validation (k), the following arguments need to be specified:
* path (str): Path to save H&E image training models.
* cnv (bool): "False" indicates that only H&E image features are used to predict tumor TMB, "True" means fusion of H&E images and clinical features to predict tumor TMB.

Then
``` shell
python main.py 
```
Other arguments such as --momentum, --step_size, --lr, --gamma, --dataset_sizes, --num_epochs and --available_policies can be specified to customize your experiments. By default results will be saved to **results**.

If you want to fuse H&E images with other features such as clinical to predict colorectal cancer TMB. Make sure all "cnv" in the code are "True".
For H&E+clinic training, look under Net.py:
```python
        # clinical feature
        self.clinical = nn.Linear(35, 35)  
        # concat
        #self.mcb = CompactBilinearPooling(128, 35, 128).cuda()
        self.mcb = CompactBilinearPooling(128, 35, 128)
        # self.concat = nn.Linear(128+55, 128)
        self.bn = nn.BatchNorm1d(128)
        self.relu = nn.ReLU(True)
        self.classifier = nn.Linear(128, 35)  
```
If your server with gpu, using "self.mcb = CompactBilinearPooling(128, 35, 128).cuda()".If your server with cpu, using "self.mcb = CompactBilinearPooling(128, 35, 128)". "35" refers to the number of fused clinical features.

Then
``` shell
python mainc.py 
```
Be careful to modify the paths in the code.


## Issues
- Please report all issues on the public forum.


