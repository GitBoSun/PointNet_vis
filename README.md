# PointNet_vis
Visualize critical points and upper bound shape of pointnet.
This work is based on "PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation"(https://arxiv.org/abs/1612.00593). Official code is on https://github.com/charlesq34/pointnet.

Here are modifications I do to implement the visualization:
1. ./models/pointnet_cls.py: I use the classification network to implement the visualization. To get the max-pooling information, I modified “get_model” to return hx and maxpool. Hx is symmetric function to map the input data to 10224 dimensional features and maxpool is the maximum of each feature among all the input points. 

2. get_file.py: I use this file to get the original data, hx and maxpool of some training samples and all points in inscribed cube in unit sphere. The former is to visualize critical points and the latter is to visualize upper bound shape and function activation. Because we cannot calculate both of them simultaneously because we should define the shape of input placeholder and their shape don’t match, I define two vis_mode(‘critical’ and ‘all’) to get them separately. 

3. vis.py: Because remote server cannot show images and our images is in 3D shape, I don’t want to save them but show them directly. This file loads ‘npz’ file gotten by get_file.py and we can visualize the points images.

Usage: 
train model: 

python train.py 

get vis file: 

python get_file.py –vis_mode ‘critical’ 

python get_file.py –vis_mode ‘all’ 

visualization: 

python vis.py
