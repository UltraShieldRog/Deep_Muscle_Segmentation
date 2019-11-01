
<!-- ### 1. Setup the dependencies (MAC & Ubuntu)
To run ..., you need to install the following packages manually beforehand


* `scikit-image-0.12.3`
* `tifffile-0.9.0`
* `scipy`
* `Cython`
* `tensorboardX`
* `pyyaml`
* `tqdm`
* `scikit-fmm==0.0.7`
```

(torch36)$ conda install scikit-image=0.12.3
(torch36)$ conda install -c conda-forge tifffile=0.9.0
(torch36)$ conda install -c anaconda scipy
(torch36)$ pip install Cython
(torch36)$ pip install tensorboardX
(torch36)$ pip install pyyaml
(torch36)$ conda install -c conda-forge tqdm
(torch36)$ pip install scikit-fmm==0.0.7

```
 -->
<!-- ### 2. Usage

**To train the model :**

```
python train.py [-h] [--config [CONFIG]] 

--config                Configuration file to use
``` -->


*<!-- *To test the model w.r.t. a dataset on custom images(s):**

```
python test.py [-h] [--model_path [MODEL_PATH]] [--dataset [DATASET]]
               [--img_path [IMG_PATH]] [--out_path [OUT_PATH]]
 
  --model_path          Path to the saved model
  --dataset             Dataset to use ['flyJanelia, camvid, ade20k etc']
  --img_path            Path of the input image
  --out_path            Path of the output segmap
```


**To visualize the loss using tensorboard:**
```
tensorboard --logdir runs
``` -->
<!-- **To organize your dataset:**
```
dataset
  => sub-dataset1
    => images (store all your original optical images eg. TIFF file)
    => labels (TIFF file converted from swc file using distance transform preprocessing)
    => ground_truth (TIFF file converted from swc file without any preprocessing)
 -->
# Deep_Muscle_Segmentation
### Paper
TBC

### 1. Setup the dependencies (MAC & Ubuntu)
To run ..., you need to install the following packages manually beforehand


* `scikit-image-0.12.3`
* `tifffile-0.9.0`
* `scipy`
* `Cython`
* `tensorboardX`
* `pyyaml`
* `tqdm`
* `scikit-fmm==0.0.7`
```
