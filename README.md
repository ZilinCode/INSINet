# INSINet
An Integrated Neighborhood and Scale Information Network for Open-Pit Mine Change Detection in High-Resolution Remote Sensing Images.
# Requirement
- Python 3.8
- Pytorch 1.10
- PIL (Pillow)
# Dataset
The dataset should consist of pairs of images (before and after) in TIFF format, with each pair containing:
A.tif and B.tif for the before and after images, respectively.
A_Neighbor.tif and B_Neighbor.tif for the corresponding neighborhood images.
label.tif for the ground truth (binary mask indicating change or no-change).

Example dataset "part_of_WHU.rar" constructed based on [the WHU change detection dataset](https://gpcv.whu.edu.cn/data/building_dataset.html) can be unzipped and used for format reference.

Organize the dataset into the following structure:
```
/data
  /dataset_name
    /train
      /<sample1>
        A.tif
        B.tif
        A_Neighbor.tif
        B_Neighbor.tif
        label.tif
      /<sample2>
        ...
    /test
      /<sample1>
        A.tif
        B.tif
        A_Neighbor.tif
        B_Neighbor.tif
      /<sample2>
      ...
```
# Working Example
1. Training the Model.

(1) Prepare the Dataset.
Make sure your dataset is structured as described above and adjust the dataset_path variable in the training script to point to your training data folder.
(2) Training Script.
To train the INSINet model, run the following command:
```
python train.py
```
Note: The model parameters are set to batch_size of 8, epochs of 200, and learning_rate of 1e-5, which you can modify according to your needs.
(3) Model Output.
At the end of training, the model's parameters will be saved in the ./model_parameter folder. You can use this saved model for inference.

2. Testing the Model.

(1) Prepare the Dataset.
Make sure your dataset is structured as described above and adjust the dataset_path variable in the testing script to point to your testing data folder.
(2) Testing Script.
To test the trained model on the test dataset, run the following command:
```
python test.py
```
Note: The test.py script loads the pre-trained model from ./model_parameter/final_model.pth. You can select the appropriate pre-trained model parameters as input.
(3) Model Output.
The model's predictions are saved as .tif files in the ./result/ folder, with each file named after the input image.

Additional Note: The example dataset "part_of_WHU.rar" is provided for initial model usage.
# Acknowledgments
The code of "MobileNet_v3" is based upon [a re-implementation](https://github.com/yichaojie/MobileNetV3) for the paper ["Searching for MobileNetV3"](https://arxiv.org/abs/1905.02244).
The example dataset "part_of_WHU.rar" constructed based on [the WHU change detection dataset](https://gpcv.whu.edu.cn/data/building_dataset.html).
Thanks for their excellent works!
# Citation
If you use this code for your research, please cite our paper.
```
@article{xie2024an,
  title={An Integrated Neighborhood and Scale Information Network for Open-Pit Mine Change Detection in High-Resolution Remote Sensing Images},
  author={Zilin Xie, Kangning Li, Jinbao Jiang, Jinzhong Yang, Xiaojun Qiao, Deshuai Yuan, and Cheng Nie},
  journal={arXiv preprint arXiv:2403.15032}, 
  pages={1-39},
  year={2024}
}
```