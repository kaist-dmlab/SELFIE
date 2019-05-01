# SELFIE: Refurbishing Unclean Samples for Robust Deep Learning

> __Publication__ </br>
> Song, H., Kim, M., and Lee, J., "SELFIE: Refurbishing Unclean Samples for Robust Deep Learning," *In Proc. 2019 Int'l Conf. on Machine Learning (ICML)*, Long Beach, California, June 2019. 

Official tensorflow implementation of **SELFIE**. Specifically, in this implementation, we tested the performance of **SELFIE** using two popular convolutional neural networks, [DenseNet [1]](http://openaccess.thecvf.com/content_cvpr_2017/html/Huang_Densely_Connected_Convolutional_CVPR_2017_paper.html) and [VGGNet [2]](https://arxiv.org/abs/1409.1556), on three simulated noisy data sets. [*Active Bias* [3]](http://papers.nips.cc/paper/6701-active-bias-training-more-accurate-neural-networks-by-emphasizing-high-variance-samples) and [*Co-teaching* [4]](http://papers.nips.cc/paper/8072-co-teaching-robust-training-of-deep-neural-networks-with-extremely-noisy-labels), which are the two state-of-the-art robust training methods, were compared with **SELFIE**.

## 1. Overview
Owing to the extremely high expressive power of deep neural networks, their side effect is to totally memorize training data even when the labels are extremely noisy. To overcome overfitting on the noisy labels, we propose a novel robust training method, which we call **SELFIE**, that trains the network on precisely calibrated samples together with clean samples. As in below Figure, it *selectively* corrects the losses of the training samples classified as *refurbishable* and combines them with the losses of clean samples to propagate backward. Taking advantage of this design, **SELFIE** effectively prevents the risk of noise accumulation from the false correction and fully exploits the training data.

<p align="center">
<img src="figures/key_idea.png " width="400"> 
</p>

## 2. Implementation
**SELFIE** requires only a simple modification in the gradient descent step. As described below, the conventional update equation is replaced with the proposed one. If you are interested in further details, read our paper.

<p align="center">
<img src="figures/update_equation.png " width="850"> 
</p>

## 2. Compared Algorithms
We compared **SELFIE** with two state-of-the-art robust training methods. We also provide the links of official/unofficial implementations for each method.
- *Active Bias* [3]: [unofficial (Tensorflow)](https://github.com/songhwanjun/ActiveBias)
- *Co-teaching* [4]: [official (Pytorch)](https://github.com/bhanML/Co-teaching) and [unofficial (Tensorflow)](https://github.com/songhwanjun/Co-teaching)

## 3. Benchmark Datasets
We evaluated the performance of **SELIFE** on *four* benchmark datasets.

| Name           | # Training Images | # Testing Images  | # Classes |  Resolution |  Link   |
| :------------: | :---------------: | :---------------: |:---------:|:----------:|:-------:|
| CIFAR-10 (clean)       | 50,000            | 10,000            | 10        |    32x32   | [link](https://drive.google.com/open?id=1ipishY18dUv7aopE36gicbNYhk9E9nHx) |
| CIFAR-100 (clean)     | 50,000            | 10,000            | 100       |    32x32   | [link](https://drive.google.com/drive/u/1/folders/1F3QE7E6-fqA1xo9wDT1VwiYRS4UYlG-X) |
| Tiny-ImageNet (clean) | 100,000           | 10,000            | 200       |    64x64   | [link](https://drive.google.com/drive/folders/1DMfyB8soRKGfR5b_MDg4uQBTg8waX5Ew?usp=sharing) |
| ANIMAL-10N (noisy) | 50,000           | 5,000            | 10       |    64x64   | [link](https://dm.kaist.ac.kr/dataset/animal-10n) |

For ease of experimentation, we provide download links for all datasets converted to the binary version. 
```
The binary version contains the files data_batch_1.bin, data_batch_2.bin, ..., as well as test_batch.bin. 
Each of these files is formatted as follows:
<id><label><depth x height x width>
...
<id><label><depth x height x width>
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
## 4. Noise Injection

## 5. Configuration

## 6. Performance

## 7. Tutorial

## 8. Reference
[1] Huang et al., 2017, Densely connected convolutional networks. In CVPR.</br>
[2] Simonyan et al., 2014, Very deep convolutional networks for large-scale image recognition. arXiv:1409.1556.</br>
[3] Chang et al., 2017, Active Bias: Training more accurate neural networks by emphasizing high variance samples. In NIPS.</br>
[4] Han et al., 2018, Co-teaching: Robust training of deep neural networks with extremely noisy labels. In NIPS.</br>
