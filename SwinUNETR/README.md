# SwinUNETR AbdomenAtlas Benchmark

Model Overview
![image](./assets/swin_unetr.png)
This repository contains the code for Swin UNETR [1,2]. Swin UNETR is the state-of-the-art on Medical Segmentation
Decathlon (MSD) and Beyond the Cranial Vault (BTCV) Segmentation Challenge dataset. In [1], a novel methodology is devised for pre-training Swin UNETR backbone in a self-supervised
manner. We provide the option for training Swin UNETR by fine-tuning from pre-trained self-supervised weights or from scratch.


# Installing Dependencies
Dependencies can be installed using:
``` bash
pip install -r requirements.txt
```

# Models

Please download the trained weights for Swin UNETR backbone (subject to update the lastest) from this <a href="https://www.dropbox.com/scl/fi/pdi87coa8ici5mqb51dli/model_swinunetrv2_s3_abdomenatlas.pt?rlkey=4bh0i1149p6hc08d8b4lbe0se&st=ccmot9ac&dl=0"> link</a>.


# Data Preparation

For training, it's needed to prepare a JSON file. 

We provide the json file that is used to train our models for AbdomenAtlas in the following <a href="https://drive.google.com/file/d/1t4fIQQkONv7ArTSZe4Nucwkk1KfdUDvW/view?usp=sharing"> link</a>.

Once the json file is downloaded, please place it in the same folder as the dataset. 
Set the correct dataset root dir under ```dataset```
Note that you need to provide the location of your dataset directory by using ```--data_dir```.

# Training

A Swin UNETR network with standard hyper-parameters for multi-organ semantic segmentation (BTCV dataset) is be defined as:

``` python
model = SwinUNETR(
    img_size=(args.roi_x, args.roi_y, args.roi_z),
    in_channels=args.in_channels,
    out_channels=args.out_channels,
    feature_size=args.feature_size,
    drop_rate=0.0,
    attn_drop_rate=0.0,
    dropout_path_rate=args.dropout_path_rate,
    use_checkpoint=args.use_checkpoint,
    use_v2=True
    )
```

The above Swin UNETR model is used for CT images (1-channel input) with input image size ```(96, 96, 96)``` and for ```10``` class segmentation outputs and feature size of  ```48```.

Using the default values for hyper-parameters, the following command can be used to initiate training using PyTorch native AMP package:

Sample: the dataroot is ./monai_benchmakr/dataset/{subject_names_folders}

## Multi-GPU Train

``` bash
python main.py --json_list=./monai_benchmark/jsons/dataset_benchmark_JHU.json --data_dir=./monai_benchmark --roi_x=96 --roi_y=96 --roi_z=96 --batch_size=1 --max_epochs=30000 --save_checkpoint true --distributed true --optim_lr=2e-4 --val_every 40 --logdir "swinunetr_s3"
```

## Inference

``` bash
python test.py --data_dir=./monai_benchmark/dataset_root_path --pretrained_model_name model_swinunetrv2_s3_abdomenatlas.pt
```

# Segmentation Output

By following the commands for evaluating `Swin UNETR` in the above, `test.py` saves the segmentation outputs
in the original spacing in a new folder based on the name of the experiment which is passed by `--exp_name`.

# Citation
If you find this repository useful, please consider citing the following papers:

```
@inproceedings{tang2022self,
  title={Self-supervised pre-training of swin transformers for 3d medical image analysis},
  author={Tang, Yucheng and Yang, Dong and Li, Wenqi and Roth, Holger R and Landman, Bennett and Xu, Daguang and Nath, Vishwesh and Hatamizadeh, Ali},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={20730--20740},
  year={2022}
}
```

# References
[1]: Tang, Y., Yang, D., Li, W., Roth, H.R., Landman, B., Xu, D., Nath, V. and Hatamizadeh, A., 2022. Self-supervised pre-training of swin transformers for 3d medical image analysis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 20730-20740).

[2]: Hatamizadeh, A., Nath, V., Tang, Y., Yang, D., Roth, H. and Xu, D., 2022. Swin UNETR: Swin Transformers for Semantic Segmentation of Brain Tumors in MRI Images. arXiv preprint arXiv:2201.01266.
