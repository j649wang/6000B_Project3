# 6000B_Project3
Task 3: Domain Adaptation.

Different x-ray devices have different image qualities and resolutions. Therefore, the model trained on the dataset from one device may fail to predict on the dataset from another device. In this situation, domain adaptation helps address the problem of differences between the two datasets.

Both datasets are needed in this task: Dataset_A works as the source and Dataset_B works as the target.

Any additional pre- or post-processing on the training set is allowed including downsampling.

You can choose to use standard classification, multi-instance learning or multi-view learning as the base model:
reference for multi-instance learning: Deep multi-instance networks with sparse label assignment for whole mammogram classification, MICCAI (3) 2017
reference for multi-view learning: High-Resolution Breast Cancer Screening with Multi-View Deep Convolutional Neural Networks, arxiv 2017

At the first step, you need to use the finetune to do supervised domain adaptation first.

At the second step, if you choose the standard classification as the base model, each team need to implement at least three of the following unspervised domain adaptation methods where labels in the training data of the target domain cannot be used; otherwise, you need to implement at least two of the abovementioned unsupervised domain adaptation methods.

MMD: Deep Domain Confusion: Maximizing for Domain Invariance
Gradient Reverse: Unsupervised Domain Adaptation by Backpropagation. ICML 2015
Adversarial Training: Adversarial Discriminative Domain Adaptation. CVPR 2017
Generate to adapt: Unsupervised Pixel-Level Domain Adaptation with Generative Adversarial Networks. CVPR 2017
Cycle GAN: Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks. ICCV 2017

Coupon will be given if you try more unspervised domain adaptation solutions than required or combine different base models.
Grading will be based on the testing accuracy, please upload the predicted results of both tasks in csv format.
The source codes should be uploaded to github.
You need to write a report to describe the details of your implementation and the report should be put in github.
