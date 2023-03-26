# TGRS_2_Multiscanning_Trans
This repository is prepared for the upcoming TGRS paper. 
# Multiscanning Strategy-based RNN-Transformer for Hyperspectral Image Classification
--------------------------------
**The conceptual idea is briefly shown below:**
<img src="https://user-images.githubusercontent.com/16319829/81180309-2b51f000-8fee-11ea-8a78-ddfe8c3412a7.png" width=50% height=50%>
--------------------------------
**The detailed structure of the proposed idea is shown below.**
![image](https://github.com/zhouweilian1904/TGRS_2_Multiscanning_Trans/blob/main/conceptual%20idea.png)

--------------------------------
**Abstract:**
The goal of hyperspectral image (HSI) classification is to assign land-cover labels to each HSI pixel in a patch-wise manner. Recently, sequential models, such as recurrent neural networks (RNN), have been developed as HSI classifiers which need to scan the HSI patch into a pixel-sequence with the scanning order first. However, RNNs have a biased ordering that cannot effectively allocate attention to each pixel in the sequence, and previous methods that use multiple scanning orders to average the features of RNNs are limited by the validity of these orders. To solve this issue, it is naturally inspired by Transformer and its self-attention to discriminatively distribute proper attention for each pixel of the pixel-sequence and each scanning order. Hence, in this study, we further develop the sequential HSI classifiers by a specially designed RNN-Transformer (RT) model to feature the multiple sequential characters of the HSI pixels in the HSI patch. Specifically, we introduce a multiscanning-controlled positional embedding strategy for the RT model to complement multiple feature fusion. Furthermore, the RT encoder is proposed for integrating ordering bias and attention re-allocation for feature generation at the sequence-level. Additionally, the spectral-spatial-based soft masked self-attention is proposed for suitable feature enhancement. Finally, an additional Fusion Transformer is deployed for scanning order-level attention allocation. As a result, the whole network can achieve competitive classification performance on four accessible datasets than other state-of-the-art methods. Our study further extends the research on sequential HSI classifiers.
