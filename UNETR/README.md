

# Installing Dependencies
Dependencies can be installed using:
``` bash
pip install -r requirements.txt
```

# Models

Please download the trained weights for UNETR backbone (subject to update the lastest) from this <a href="https://www.dropbox.com/scl/fi/bmmd7xrfb8grdmgkrtk2v/model_unetr_0530_s2.pt?rlkey=3q69tmoyz6qph8aznbm66umou&st=7urhe3az&dl=0"> link (TBA)</a>.


# Data Preparation

For training, it's needed to prepare a JSON file. 

We provide the json file that is used to train our models for AbdomenAtlas in the following <a href="https://drive.google.com/file/d/1t4fIQQkONv7ArTSZe4Nucwkk1KfdUDvW/view?usp=sharing"> link</a>.

Once the json file is downloaded, please place it in the same folder as the dataset. 
Set the correct dataset root dir under ```dataset```
Note that you need to provide the location of your dataset directory by using ```--data_dir```.

# Training

A Swin UNETR network with standard hyper-parameters for multi-organ semantic segmentation (BTCV dataset) is be defined as:

``` python
    model = UNETR(
        in_channels=1,
        out_channels=10,
        img_size=(args.roi_x, args.roi_y, args.roi_z),
        feature_size=32,
        hidden_size=768,
        mlp_dim=3072,
        num_heads=16,
        pos_embed='conv',
        norm_name='instance',
        conv_block=True,
        res_block=True,
        dropout_rate=0.0)
```

The above Swin UNETR model is used for CT images (1-channel input) with input image size ```(96, 96, 96)``` and for ```10``` class segmentation outputs and feature size of  ```48```.

Using the default values for hyper-parameters, the following command can be used to initiate training using PyTorch native AMP package:

Sample: the dataroot is ./monai_benchmakr/dataset/{subject_names_folders}

## Multi-GPU Train

``` bash
python main.py --json_list=./monai_benchmark/jsons/dataset_benchmark_JHU.json --data_dir=./monai_benchmark --roi_x=96 --roi_y=96 --roi_z=96 --batch_size=1 --max_epochs=30000 --save_checkpoint true --distributed true --optim_lr=2e-4 --val_every 40 --logdir "unetr_s3"
```

## Inference

``` bash
python test_unetr.py --data_dir=./monai_benchmark --pretrained_model_name model_unetr_s2.pt
```


# Citation
If you find this repository useful, please consider citing the following papers:

```
@inproceedings{hatamizadeh2022unetr,
  title={Unetr: Transformers for 3d medical image segmentation},
  author={Hatamizadeh, Ali and Tang, Yucheng and Nath, Vishwesh and Yang, Dong and Myronenko, Andriy and Landman, Bennett and Roth, Holger R and Xu, Daguang},
  booktitle={Proceedings of the IEEE/CVF winter conference on applications of computer vision},
  pages={574--584},
  year={2022}
}
```
