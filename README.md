# CycleGAN_Monet

## Members

Ana Vera Vazquez (aver@itu.dk), 

Julie Hinge (juhi@itu.dk), 

Carolina Brañas Soria (cabra@itu.dk).

## Central problem

We aimed to learn and analyze the main strengths and challenges of using CycleGANs to generate a Monet-styled painting based on an input photography.

## Domain

The model belongs to the unpaired image-to-image translation domain. As explained in the original CycleGAN paper, the “primary focus is learning the mapping between two image collections, rather than between two specific images, by trying to capture correspondences between higher-level appearance structures. Therefore, our method can be applied to other tasks, such as painting → photo, object transfiguration, etc. where single sample transfer methods do not perform well” (https://arxiv.org/abs/1703.10593). 

The identification of higher-level appearance structures makes it great to train CycleGANs to learn the features of Monet paintings and apply them to real images.

## Data characteristics

The dataset of choice contains four directories: `monet_tfrec`, `photo_tfrec`, `monet_jpg`, and `photo_jpg`. The `monet_tfrec` and `monet_jpg` directories contain the same painting images, and the `photo_tfrec` and `photo_jpg` directories contain the same photos.

The `monet` directories contain Monet paintings. We used these images to train our model.

The `photo` directories contain photos. We will add Monet-style to these images. Other photos outside of this dataset can be transformed but did not experiment on that based on time constrains.

Files: 

- **monet_jpg** - 300 Monet paintings sized 256x256 in JPEG format
- **monet_tfrec** - 300 Monet paintings sized 256x256 in TFRecord format
- **photo_jpg** - 7028 photos sized 256x256 in JPEG format
- **photo_tfrec** - 7028 photos sized 256x256 in TFRecord format

Link to the dataset: https://www.kaggle.com/competitions/gan-getting-started/data

## Central method and CycleGAN architecture

### Architecture

Our method of choice is CycleGAN for image-to-image translation with unpaired training data. The model aims to learn characteristics from images (in this case, Monet paintings) and translate them to other images.

We implemented the CycleGAN architecture following the Pix2Pix (paired image-to-image translation) architecture. Unlike Pix2Pix models, Cycle GANs use two generators and two discriminators to achieve results. Generators and discriminators use instance normalization and these are their tasks:

- **Generator:** It transforms input images into Monet's artistic style and viceversa.
- **Discriminator:** Differentiates between real Monet images and generated images.


#### Architecture Summary

- **GAN 1**: Translates real images (collection 1) to Monet (collection 2).
- **GAN 2**: Translates Monet (collection 2) to real images (collection 1).

We can summarize the generator and discriminator models from GAN 1 as follows:

- **Generator Model 1:**
    - **Input**: Takes real images (collection 1).
    - **Output**: Generates Monet (collection 2).
- **Discriminator Model 1**:
    - **Input**: Takes Monet from collection 2 and output from Generator Model 1 (fake Monet).
    - **Output**: Likelihood of image is from collection 2.
- **Losses**:
    - Cycle consistency loss

Similarly, we can summarize the generator and discriminator models from GAN 2 as follows:

- **Generator Model 2**:
    - **Input**: Takes Monet (collection 2).
    - **Output**: Generates real images (collection 1).
- **Discriminator Model 2**:
    - **Input**: Takes real images from collection 1 and output from Generator Model 2 (fake real images).
    - **Output**: Likelihood of image is from collection 1.
- **Losses**:
    - Cycle consistency loss

So far with this architecture, the models are sufficient for generating plausible images in the target domain but are not translations of the input image. This is when the CycleGAN model differs from the Pix2Pix model:  Pix2Pix uses a combination of adversarial loss and L1 loss between the generated and target images, while CycleGAN incorporates cycle-consistency loss in addition to adversarial loss.

### Losses

1. **Adversarial loss**: to improve the performance of both generator and discriminator. Generator aims to minimize this loss against its corresponding Discriminator that tries to maximize it. But adversarial losses alone cannot guarantee that the learned function can map an individual input xi to a desired output yi. To solve this problem, the authors used the cycle consistency loss.

2. **Cycle consistency loss:**  The CycleGAN encourages cycle consistency by adding an additional loss to measure the difference between the generated output of the second generator and the original image, and the reverse. This acts as a regularization of the generator models, guiding the image generation process in the new domain toward image translation.. it calculates the L1 loss between the original image and the final generated image, which should look same as original image. It is calculated in two directions:

    - Forward Cycle Consistency: Domain-B -> **Generator-A** -> Domain-A -> **Generator-B** -> Domain-B
    - Backward Cycle Consistency: Domain-A -> **Generator-B** -> Domain-B -> **Generator-A** -> Domain-A

The cycle consistency loss also helps solving the mode collapse problem, in which all input images map to the same output image and the optimization fails to make progress. This type of loss helps reducing the space of possible mapping functions, so the model learns to differenciate style instead of other features.

3. **Identity loss:** It encourages the generator to preserve the color composition between input and output. This is done by providing the generator an image of its target domain as an input and calculating the L1 loss between input and the generated images.
    - Domain-A -> **Generator-A** -> Domain-A
    - Domain-B -> **Generator-B** -> Domain-B


### Training and hyperparameters

We split the original data into training (80%) and test (20%):

```
Shape of X_train_A: (5630, 256, 256, 3)
Shape of X_test_A: (1408, 256, 256, 3)
Shape of X_train_B: (240, 256, 256, 3)
Shape of X_test_B: (60, 256, 256, 3)
```

We initially trained the model for 10 epochs, with a batch size of 1, learning rates of 2e-4 and Adam optimizers for both the discriminators and generators. CycleGANs are known for being very heavy and time consuming to train, so we used the U-Cloud cluster as well as the available 30 hours of Kaggle GPU for our training process.

### Evaluation

Style is very difficult to define and hence to measure. Measuring how well style is captured by the model was a challenging tasks, and despite being plenty of quantitative solutions out there, most of them are inaccurate or just less reliable than the human eye. For this particular task and due to time constrains, we decided to evaluate the results visually, analyzing if the Monet style was present in the output images.

~~We also plot the training losses to help understand if the model is improving or not, and to decide if it is needed to increase the initial learning rate.~~

## Key experiments and results

We first tested how the model performed without any training. Here are the results:

![Alt text](/readme_images/cyclegan1.png?raw=true "Optional Title")

![Alt text](/readme_images/cyclegan2.png?raw=true "Optional Title")

### PatchGAN Discriminator

The discriminators use PatchGAN to determine if the input image is a real Monet or a real image. The PatchGAN discriminator tries to classify if each 𝑁×𝑁 patch in an image is real or fake. This discriminator is run convolutionally across the image, averaging all responses to provide the ultimate output of 𝐷. Such a discriminator effectively models the image as a Markov random field, assuming independence between pixels separated by more than a patch diameter. It can be understood as a type of texture/style loss.

The image below represents the outputs of the Monet discriminator PatchGAN ~~before training (?)~~  . Red tones mean that the discriminator classifies the patch as pertaining to a non-Monet image, while blue tones mean that the discriminator classifies the patch as pertaining to a Monet image:

![Alt text](/readme_images/cyclegan3.png?raw=true "Optional Title")

### Data Augmentation

As there was a class imbalance between the Monet training pictures (240) in comparison to real images (5630), we suspected that this could result in poor performance in generating high-quality monet-style images. To solve this issue, we performed data augmentation on the Monet images training set, which consisted in applying:

- Random jitter in 50 images
- Vertical flip in 50 images
- Random crop in 50 images
- Saturation in 25 images
- Brightness in 25 images

As a result, we generated an extra 200 Monet training images, resulting in a total of 440 Monet training images.

~~We also experimented with other parameters like the learning rate and number of epochs.~~

The results can be visualized below these lines:

[ Insert simple model pictures here ]

[ Insert data augmentation model pictures here ]

[ Tuned hyperparameters model pictures here ]

In addition, we would have liked to implement a modified ResNet generator used in the original CycleGAN paper, instead of the U-Net generator used here.

## Results and Discussion

### Model 1: 10 epochs with original data, lr = 0.004

[ Summarize results here ]

### Model 2: 10 epochs with augmented data, lr = 0.004

Throughout the epochs, we test the results (turning a real image into a Monet painting) on one sample image. We notice that the generated image slowly becomes less blurry throughout the epochs. 

### Model 3: 10 epochs with augmented data, lr = 0.006

[ Summarize results here ]

### Model 4: 20 epochs with augmented data, lr = 0.004

[ Summarize results here ]


### Results

Overall, we noticed that the model performs slightly better with images of nature. This might happen because the training data contains plenty of nature images, whereas it does not contain many images of humans, for example. 

Analyzing some of the PatchGAN patch by patch classifications, we also notice that the plain sky is also hard to classify for the discriminator as either a Monet or a real image patch [ INSERT IMAGES ]

We found it quite challenging to understand the architecture as it is quite complex, as well as to train the model, which requires plenty of time and resources (hence, we used the U-Cloud Cluster for training). Despite all these challenges, we were satisfied with the initial results of our models. CycleGANs are very unstable models that can often have issues converging and not provide any feasible results

With more time, we would have liked to:

- implement a different model using a ResNet modified generator (like in the original CycleGAN paper),
- increase the number of epochs to improve performance,
- apply learning rate decay after 100 epochs like in the original paper,
- further analyse the results and perform different data augmentation techniques based on color,
- develop quantitative evaluation metrics to analyze how well the Monet style is captured and applied to new images by our model.

## References

CycleGAN original paper.

Kaggle data.
