# SELFIE: Refurbishing Unclean Samples for Robust Deep Learning

> __Publication__ </br>
> Song, H., Kim, M., and Lee, J., "SELFIE: Refurbishing Unclean Samples for Robust Deep Learning," *In Proc. 2019 Int'l Conf. on Machine Learning (ICML)*, Long Beach, California, June 2019. [[link]](http://proceedings.mlr.press/v97/song19b.html)

Official tensorflow implementation of **SELFIE**. Specifically, in this implementation, we tested the performance of **SELFIE** using two popular convolutional neural networks, [DenseNet [1]](http://openaccess.thecvf.com/content_cvpr_2017/html/Huang_Densely_Connected_Convolutional_CVPR_2017_paper.html) and [VGGNet [2]](https://arxiv.org/abs/1409.1556), on not only three simulated noisy datasets but also a real-world dataset. [*Active Bias* [3]](http://papers.nips.cc/paper/6701-active-bias-training-more-accurate-neural-networks-by-emphasizing-high-variance-samples) and [*Co-teaching* [4]](http://papers.nips.cc/paper/8072-co-teaching-robust-training-of-deep-neural-networks-with-extremely-noisy-labels), which are the two state-of-the-art robust training methods, were compared with **SELFIE**.

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
We compared **SELFIE** with default and two state-of-the-art robust training methods. We also provide the links of official/unofficial implementations for each method (The three algorithms are included in our implementation).
- *Default*: Training method without any processing for noisy label.
- *Active Bias* [3]: [unofficial (Tensorflow)](https://github.com/songhwanjun/ActiveBias)
- *Co-teaching* [4]: [official (Pytorch)](https://github.com/bhanML/Co-teaching) and [unofficial (Tensorflow)](https://github.com/songhwanjun/Co-teaching)

## 3. Benchmark Datasets
We evaluated the performance of **SELIFE** on *four* benchmark datasets. Here, ANIMAL-10N data set is our proprietary real-world noisy dataset of human-labled online images for 10 confusing animals. Please note that, in ANIMAL-10N, noisy labels were injected *naturally* by human mistakes, where its noise rate was estimated at *8%*.

| Name (clean or noisy)    | # Training Images | # Testing Images  | # Classes |  Resolution |  Link   |
| :------------: | :---------------: | :---------------: |:---------:|:----------:|:-------:|
| CIFAR-10 (clean)       | 50,000            | 10,000            | 10        |    32x32   | [link](https://drive.google.com/a/dm.kaist.ac.kr/file/d/1ipishY18dUv7aopE36gicbNYhk9E9nHx/view?usp=sharing) |
| CIFAR-100 (clean)     | 50,000            | 10,000            | 100       |    32x32   | [link](https://drive.google.com/a/dm.kaist.ac.kr/file/d/1vhYTpKzD4Y64SkcQgoMfHOg03ZgavwRw/view?usp=sharing) |
| Tiny-ImageNet (clean) | 100,000           | 10,000            | 200       |    64x64   | [link](https://drive.google.com/a/dm.kaist.ac.kr/file/d/1tR-fW1htexV-H9kMmpIQfF-KNZJFB7ZB/view?usp=sharing) |
| ANIMAL-10N (noisy) | 50,000           | 5,000            | 10       |    64x64   | [link](https://dm.kaist.ac.kr/datasets/animal-10n) |

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
Except ANIMAL-10N dataset, since all datasets are clean, we artifically corrupted CIFAR-10, CIFAR-100 and Tiny-ImageNet datasets using two typical methods such that the true label *i* is flipped into the corrupted label *j*: *i)* **Pair Noise** and *ii)* **Symmetry Noise**. Below figures show the example of the noise transition matrix for each type.
<p align="center">
<img src="figures/noise_type.png " width="550"> 
</p>

As for **real-world noisy** ANIMAL-10N dataset, the noise rate of training data is found at **8%** by the corss-validation with grid search (See Appendix B).

## 5. Environment and Configuration
- Python 3.6.4
- Tensorflow-gpu 1.8.0 (pip install tensorflow-gpu==1.8.0)
- Tensorpack (pip install tensorpack)

In our paper, for the evaluation, we used a *momentum* of 0.9, a *batch size* of 128, a *dropout* of 0.2, and *batch normalization*. For training schedule, we trained the network for *100 epochs* and used an *initial learning rate* of 0.1, which was divided by 5 at 50% and 75% of the toral number of epochs. 
As for the algorithm hyperparameters, we fixed *restart* to 2 and used the best uncertainty threshold *epsilon* = 0.05, history length *q* = 15, which were obtained using the grid search (See Section 4.5 in our paper).

## 6. Performance
We trained DenseNet (L=25, k=12) and VGG-19 on the four benchmark datasets. The detailed anaysis on the evalutaion is discussed in our paper.

#### 6.1 Synthetic Noise (CIFAR-10/100, Tiny-ImageNet)
- DenseNet (L=25, k=12) on varying noise rates.
<p align="center">
<img src="figures/synthetic_evaluation.png " width="670"> 
</p>

#### 6.2 Real-World Noise (ANIMAL-10N)
- The noise rate of ANIMAL-10N is estimated at 8%.
<p align="center">
<img src="figures/realistic_evaluation.png " width="420"> 
</p>

## 7. How to Run
- Dataset download:
   ```
   Download our datasets (binary format) and place them into *SELFIE/dataset/xxxxx*.
   (e.g., SELFIE/dataset/CIFAR-10)
   ```
- Algorithm parameters
   ```
    -gpu_id: gpu number which you want to use (only support single gpu).
    -data: dataset in {CIFAR-10, CIFAR-100, Tiny-ImageNet, ANIMAL-10N}.
    -model_name: model in {VGG-19, DenseNet-10-12, DenseNet-25-12, DenseNet-40-12}.
    -method_name: method in {Default, ActiveBias, Coteaching, SELFIE}.
    -noise_type: synthetic noise type in {pair, symmetry, none}, none: do not inject synthetic noise.
    -noise_rate: the rate which you want to corrupt (for CIFAR-10/100, Tiny-ImageNet) or the true noise rate of dataset (for ANIMAL-10N).
    -log_dir: log directory to save the training/test error.
   ```
   
- Algorithm configuration

   Data augmentation and distortion are not applied, and training paramters are set to:
   ```
   Training epochs: 100
   Batch size: 128
   Learning rate: 0.1 (divided 5 at the approximately 50% and approximately 75% of the total number of epochs)
   ```

- Running commend
   ```python
   python main.py gpu_id data model_name method_name noise_type noise_rate log_dir
   
   # e.g. 1., train DenseNet (L=25, k=12) on CIFAR-100 with pair noise of 40%.
   # python main.py 0 CIFAR-100 DenseNet-25-12 SELFIE pair 0.4 log/CIFAR-100/SELFIE
   
   # e.g. 2., train DenseNet (L=25, k=12) on ANIMAL-10N with real-world noise of 8%
   # python main.py 0 ANIMAL-10N DenseNet-25-12 SELFIE none 0.08 log/ANIMAL-10N/SELFIE
   ```

- Detail of log file
   ```
   log.csv: generally, it saves training loss/error and test loss/error.
    - format : epoch, training loss, training error, test loss, test error
   However, Coteaching uses two network, so format is slightly different.
    - format : epoch, training loss (network1), training error (notwork1), training loss (network2), training error (network2), test loss (notwork1), test error (network1), test loss (network2), test error (network2)
   ```
   
## 8. Reference
[1] Huang et al., 2017, Densely connected convolutional networks. In CVPR.</br>
[2] Simonyan et al., 2014, Very deep convolutional networks for large-scale image recognition. arXiv:1409.1556.</br>
[3] Chang et al., 2017, Active Bias: Training more accurate neural networks by emphasizing high variance samples. In NIPS.</br>
[4] Han et al., 2018, Co-teaching: Robust training of deep neural networks with extremely noisy labels. In NIPS.</br>

## 9. Contact
Hwanjun Song (songhwanjun@kaist.ac.kr); Minseok Kim (minseokkim@kaist.ac.kr); Jae-gil Lee (jaegil@kaist.ac.kr)
