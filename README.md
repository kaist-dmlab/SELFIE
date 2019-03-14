# SELFIE: Refurbishing Unclean Samples for Robust Deep Learning

> __Publication__ </br>
> Song, H., Kim, M., and Lee, J., "SELFIE: Refurbishing Unclean Samples for Robust Deep Learning," *In Proc. 2019 Int'l Conf. on Machine Learning (ICML)*, Long Beach, California, June 2019. 

Official tensorflow implementation of **SELFIE**. Specifically, in this implementation, we tested the performance of **SELFIE** using two popular convolutional neural networks, DenseNet [(Huang et al., CVPR 2017)](http://openaccess.thecvf.com/content_cvpr_2017/html/Huang_Densely_Connected_Convolutional_CVPR_2017_paper.html) and VGG [(Simonyan et al., CVPR 2015)](https://arxiv.org/abs/1409.1556), on three simulated noisy data sets. *Active Bias* [(Chang et al., NIPS 2017)](http://papers.nips.cc/paper/6701-active-bias-training-more-accurate-neural-networks-by-emphasizing-high-variance-samples) and *Co-teaching* [(Han et al., NIPS 2018)](http://papers.nips.cc/paper/8072-co-teaching-robust-training-of-deep-neural-networks-with-extremely-noisy-labels), which are the two state-of-the-art robust training methods, were compared with **SELFIE**.

## 1. Overview
Owing to the extremely high expressive power of deep neural networks, their side effect is to totally memorize training data even when the labels are extremely noisy. To overcome overfitting on the noisy labels, we propose a novel robust training method called. **SELFIE** trains the network on precisely calibrated samples together with clean samples. As in below Figure, it *selectively* correct the losses of the training samples classified as *refurbishable* and combine them with the losses of clean samples to propagate backward.Taking advantage of this design, **SELFIE** effectively prevents the risk of noise accumulation from the false correction and fully exploits the training data.

<p align="center">
<img src="figures/key_idea.png " width="450"> 
</p>

## 2. Algorithms

## 3. Data Sets
| Name           | # Training Images | # Testing Images  | # Classes |  Link   |
| :------------: | :---------------: | :---------------: |:---------:|:-------:|
| CIFAR-10       | 50,000            | 10,000            | 10        | [link](https://drive.google.com/drive/folders/1q8zYWwB5gOMJm35XgcMd0zpxwmgEFlCi?usp=sharing) |
| CIFAR-100      | 50,000            | 10,000            | 100       | [link](https://drive.google.com/drive/folders/1gMikxSdScmQxGxfjwtgOXYeGvWKcx8eN?usp=sharing) |
| Tiny-ImageNet  | 100,000           | 10,000            | 200       | [link](https://drive.google.com/drive/folders/1DMfyB8soRKGfR5b_MDg4uQBTg8waX5Ew?usp=sharing) |

For ease of experimentation, we provide download links for all datasets converted to the binary version. 
```
The binary version contains the files data_batch_1.bin, data_batch_2.bin, ..., as well as test_batch.bin. 
Each of these files is formatted as follows:
<id><label><depth x width x height>
...
<id><label><depth x width x height>
```

The reading procedure is similar to that of a popular [CIFAR-10 tutorial](https://github.com/tensorflow/models/blob/master/tutorials/image/cifar10/cifar10_input.py).
```python
# You can read our bianry files as below: 
ID_BYTES = 4
LABEL_BYTES = 4
RECORD_BYTES = ID_BYTES + LABEL_BYTES + width * height * depth
reader = tf.FixedLengthRecordReader(record_bytes=RECORD_BYTES)
file_name, value = reader.read(filename_queue)
byte_record = tf.decode_raw(value, tf.uint8)
image_id = tf.strided_slice(byte_record, [0], [ID_BYTES])
image_label = tf.strided_slice(byte_record, [ID_BYTES], [ID_BYTES + LABEL_BYTES])
array_image = tf.strided_slice(byte_record, [ID_BYTES + LABEL_BYTES], [RECORD_BYTES])
depth_major_image = tf.reshape(array_image, [depth, height, width])
record.image = tf.transpose(depth_major_image, [1, 2, 0])
```

## 4. Configuration

## 5. How to run

## 6. Tutorial
