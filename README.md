# Multiscanning Strategy-based RNN-Transformer for Hyperspectral Image Classification. (resubmit major revision)

--------------------------------
Weilian Zhou, Sei-ichiro Kamata, Haipeng Wang, Xi Xue

Feel free to contact us if there is anything we can help. Thanks for your support!

zhouweilian1904@akane.waseda.jp, kam@waseda.jp

--------------------------------
**The detailed structure of the proposed idea is shown below.**
![image](https://github.com/zhouweilian1904/TGRS_2_Multiscanning_Trans/blob/main/conceptual%20idea.png)

--------------------------------
**Abstract:**

The goal of hyperspectral image (HSI) classification is to assign land-cover labels to each HSI pixel in a patch-wise manner. Recently, sequential models, such as recurrent neural networks (RNN), have been developed as HSI classifiers which need to scan the HSI patch into a pixel-sequence with the scanning order first. However, RNNs have a biased ordering that cannot effectively allocate attention to each pixel in the sequence, and previous methods that use multiple scanning orders to average the features of RNNs are limited by the validity of these orders. To solve this issue, it is naturally inspired by Transformer and its self-attention to discriminatively distribute proper attention for each pixel of the pixel-sequence and each scanning order. Hence, in this study, we further develop the sequential HSI classifiers by a specially designed RNN-Transformer (RT) model to feature the multiple sequential characters of the HSI pixels in the HSI patch. Specifically, we introduce a multiscanning-controlled positional embedding strategy for the RT model to complement multiple feature fusion. Furthermore, the RT encoder is proposed for integrating ordering bias and attention re-allocation for feature generation at the sequence-level. Additionally, the spectral-spatial-based soft masked self-attention is proposed for suitable feature enhancement. Finally, an additional Fusion Transformer is deployed for scanning order-level attention allocation. As a result, the whole network can achieve competitive classification performance on four accessible datasets than other state-of-the-art methods. Our study further extends the research on sequential HSI classifiers.

--------------------------------
**Datasets:**

We have uploaded several datasets: https://drive.google.com/drive/folders/1IQxuz4jpwf6goB_2ZwVLEZz1ILScymdO?usp=share_link
1. Indian Pines, 
2. PaviaU, 
3. PaviaC, 
4. Botswana, 
5. Houston 2013, 
6. KSC, 
7. Mississippi_Gulfport, 
8. Salinas, 
9. Simulate_database, 
10. Augmented_IndianPines, 
11. Augmented_Mississippi_Gulfport, 
12. Augmented_PaviaU
13. The disjoint datasets (IP, PU, HU) can be referred in https://github.com/danfenghong/IEEE_TGRS_SpectralFormer.


--------------------------------
**How to use:**

you can find and add some arguments in *main.py* for your own testing.

For example:

python main.py --model multiTrans  --dataset IndianPines --training_sample 0.1 --cuda 0 --epoch 200 --batch_size 100 --class_balancing --patch_size 9 --flip_augmentation

--------------------------------
**Models:**

In the *model.py*, we have implemented many types of different designs for HSI classification. You can try it with your debug becasuse we are still modifying them. There may exist some mistakes. And also, we are sorry for the mess of the codes.

The first version of our model is created in *TGRS_2.py*. We will update the newest one after the paper acceptance.

--------------------------------
**Compared methods**
1. General ViT: https://github.com/lucidrains/vit-pytorch

2. SpeFormer: https://github.com/danfenghong/IEEE_TGRS_SpectralFormer

3. 1DCT: https://github.com/lironui/Double-Branch-Dual-Attention-Mechanism-Network + https://github.com/lucidrains/vit-pytorch

4. SST: https://www.researchgate.net/publication/349966825_The_demo_of_Spatial-Spectral_Transformer_for_Hyperspectral_Image_Classification

5. SSFTT: https://github.com/zgr6010/HSI_SSFTT

6. SAT: https://github.com/luuuyi/CBAM.PyTorch  +  https://github.com/lucidrains/vit-pytorch

7. 3D-ANAS: https://github.com/hkzhang91/3D-ANAS

8. CasRNN: https://github.com/RenlongHang/CasRNN

9. Multi-LSTM: https://github.com/zhouweilian1904/TGRS_1_Multiscanning_RNN



--------------------------------
**Env settings:**

Pytorch:1.11.0

Cuda:11.7

Others: you can direct install the newest version of other packages through "pip install xxx".






