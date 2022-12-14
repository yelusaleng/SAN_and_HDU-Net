# SAN_and_HDU-Net
Official repository for "Image splicing forgery detection by combining synthetic adversarial networks and hybrid dense U‐net based on multiple spaces"  
If you think the repository is helpful, give me a star to let me know.

## Requirements
1. Python: 3.7.0
2. CUDA: 11.4
3. Pytorch: 1.8.0

## Data Preparation
1. For SAN:  
   Merge CASIA v2.0 and Forensics, and put the combination dataset (1891 images) into the subdirs `/img/` and `/mask/`.
    ```shell
    HDU-Net/
    ├── ...
    └── SAN/
        ├── img/
        └── mask/
    ```
    Generate a dataset that I call $SF-Data$ (82608 images). You can download $SF-Data$ via the link <https://drive.google.com/file/d/1IoG78dAcxyw5fRPo1DjisKUoykJQTs2_/view?usp=sharing>.
    
2. For HDU-Net:  
   To generate edge information according to the subdir `/mask/`， run
   ```shell
   python SAN/generate_edge.py
   ```

## Getting Started
### Training in Command Line
1. For SAN:  
   If you want to retrain SAN, run
   ```shell
   python SAN/train.py
   ```
 
2. For HDU-Net:  
   To train HDU-Net, run
   ```shell
   python train.py
   ```
You should change diverse parameters in `options.py`
   
### Evaluation in Command Line
1. For SAN:  
    I provide a well-trained model weight `best_model_for_SAN.pth`.  
    You can download the weights via <https://drive.google.com/file/d/1Qbn3kCxwMA7r-VQ0mpnXaI1tKKetPn7n/view?usp=sharing>, and put it into the subdir `/SAN/`.  
    You can use it to generate dataset based on other datasets like COCO, etc. I have converted the multi-label annotations "train2017" in COCO to binary mask. I are hesitating to upload the dataset since it is too large.  
    After running the following codes, you should change the path of dataset in `options_GAN.py`.  
    Note that the well-trained weight only accept binary mask.
   ```shell
   python SAN/generate_data.py
   ```

2. For HDU-Net:  
   You can download the weights of HDU-Net via <https://drive.google.com/file/d/1XDMZdGzxSvs22j5uKx6Ywm92JlgFv_iw/view?usp=sharing>.  
   Run  
   ```shell
   python inference.py
   ```

## Citation
If you find this project useful for your research, please use the following BibTeX entry.
```shell
@article{Wei2022ImageSF,
  title={Image splicing forgery detection by combining synthetic adversarial networks and hybrid dense U‐net based on multiple spaces},
  author={Yang Wei and Jianfeng Ma and Zhuzhu Wang and Bin Xiao and Wenying Zheng},
  journal={International Journal of Intelligent Systems},
  year={2022}
}
```

## License
This project is released under the MIT license.

## Contact
Contact yale ywei9395@gmail.com for any further information.
