

# Installing Dependencies
Dependencies can be installed using:
``` bash
pip install -r requirements.txt
```

# Models

Please download the trained weights for UNEST backbone (subject to update the lastest) from this <a href="https://www.dropbox.com/scl/fi/e1468cx8tniulza0xben9/model_unest_lowerLR_s2.pt?rlkey=mi7yzplrl4ufegakozakunxcs&st=m6tuqccu&dl=0"> link </a>.


# Data Preparation

For training, it's needed to prepare a JSON file. 

We provide the json file that is used to train our models for AbdomenAtlas in the following <a href="https://drive.google.com/file/d/1t4fIQQkONv7ArTSZe4Nucwkk1KfdUDvW/view?usp=sharing"> link</a>.

Once the json file is downloaded, please place it in the same folder as the dataset. 
Set the correct dataset root dir under ```dataset```
Note that you need to provide the location of your dataset directory by using ```--data_dir```.

# Training

A Swin UNETR network with standard hyper-parameters for multi-organ semantic segmentation (BTCV dataset) is be defined as:

``` python
    model = UNesT(
        in_channels=args.in_channels,
        out_channels=args.out_channels,
    )
```

The above Swin UNETR model is used for CT images (1-channel input) with input image size ```(96, 96, 96)``` and for ```10``` class segmentation outputs and feature size of  ```48```.

Using the default values for hyper-parameters, the following command can be used to initiate training using PyTorch native AMP package:

Sample: the dataroot is ./monai_benchmakr/dataset/{subject_names_folders}

## Multi-GPU Train

``` bash
python main.py --json_list=./monai_benchmark/jsons/dataset_benchmark_JHU.json --data_dir=./monai_benchmark --roi_x=96 --roi_y=96 --roi_z=96 --batch_size=1 --max_epochs=30000 --save_checkpoint true --distributed true --optim_lr=2e-4 --val_every 40 --logdir "unest_s3"
```

## Inference

``` bash
python test_unest.py --data_dir=./monai_benchmark --pretrained_model_name model_unest_lowerLR_s2.pt
```


# Citation
If you find this repository useful, please consider citing the following papers:

```
@article{yu2023unest,
  title={UNesT: local spatial representation learning with hierarchical transformer for efficient medical segmentation},
  author={Yu, Xin and Yang, Qi and Zhou, Yinchi and Cai, Leon Y and Gao, Riqiang and Lee, Ho Hin and Li, Thomas and Bao, Shunxing and Xu, Zhoubing and Lasko, Thomas A and others},
  journal={Medical Image Analysis},
  volume={90},
  pages={102939},
  year={2023},
  publisher={Elsevier}
}
```
