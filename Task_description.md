# Homework 2: Face recognition
*Intoduction to computer vision*

**Deadline:** November 12th 23:55
**Submission format:** Python Scripts, Jupyter Notebook, C++ scripts, trained model (in .pth data format or any other that can be load), all files required to run your code submit as well. 
If you submit Python scripts create *main.py* from where training and testing could be run. If you submit Jupyter Notebook make sure *run all* is working.
**Test data:** [download](https://www.dropbox.com/s/loz4ijaxcor3l5l/test_set.csv?dl=1)
**Dataset:** [download](https://www.dropbox.com/s/7qdm7ptr53nknr8/dataset.zip?dl=1)

*Note: In your solution you are not obliged to use pytorch, you can use any other framework to build your solution. Code presented in this description might be helpful but not required to use.*

## Grading policy
Total you can get 5 points + 2 bonus point.
- 1 point for creating minibatches of images
- 1 point for assembling triplet loss
- 1 ponts for concatenating two models together VGG19 (or other) + own layers
- 1 point for reporting training loss and test set accuracy every epoch during training
- 1 point for reporting accuracy on test set after training completed
- 2 bonus points for appropriate selection of threshold T and $\alpha$ using cross validation (meaning you have to create validation set)
- **Update** 2 bonus points for trying loss function from advanced topics section

*Note: Code quality will be evaluated. Maximum -5% for bad code.*
**Update** *Note: Accuracy of the model will be evaluated. Maximum -5% for accuracy less than 70%.*
**Update** 2 bonus points maximum, i.e if you get greater than 7 points it will be equal to 7.

## Cheating policy
0 mark for copying within the group and from the Internet. 

## Unconstrained classification

The problem of unconstrained classification does not assume the finite number of classes. Unconstrained classification problems should have a way of making a decision when the set of possible classes is not completely defined.

The possible solution to this problem is to define a continuous space and dedicate different region of this space to specific classes. This way, the new classes can be added on the fly, since all we need is to decide what a particular region of space represents. As you can guess this approach is related to the problems of clustering and nearest neighbor search.

But simple nearest neighbor classification will not work because it will assign a class even when the new sample is really far from any of the centroids of any of the known clusters. The simplest solution is to add some threshold. The classification rule now is
$$
\begin{equation}
  \hat{c}=\begin{cases}
    \underset{c\in C}{\operatorname{argmax}}R(x, c), & \text{if } \exists c \in C: R(x, c) > T,\\
    \emptyset, & \text{otherwise}.
  \end{cases}
\end{equation}
$$

where T is the similarity threshold. In this classification rule is we leave ourselves the opportunity to refuse to assign any class, in case our confidence level is not sufficient. We make this decision by requiring the correct class to have similarity of at least $T$. 

## Triplet Loss (Margin Loss)

For the task of face classification, we want to find an embedding of a face that captures special facial properties so that it is easy to tell different faces apart by looking at some similarity measure. In other words, we want an embedding function $f$ that projects an image $x$ into the embeddings space, where different faces are at least as far from each other as the threshold value $T$. Since it is hard to come up with such a function $f$, we try to learn it. The only thing we care about is the interpretation of the distance between embeddings, and this will be the only criteria that we enforce using the optimization loss. Given an image, we call it an anchor, we want to minimize the distance between other images of the same face, call it positive examples, and we want to maximize the difference with other faces. Formally

$$
||f(anc) - f(pos)||^2 + \alpha \leq ||f(anc) - f(neg)||^2
$$

where $anc$ is the anchor image, $pos$ and $neg$ are positive and negative images correspondingly. If we remember the notion of similarity function $R$ from the previous section, for this loss it is defined as follows

$$
R(x, c) = - ||f(c) - f(x)||^2
$$
where maximum similarity is 0, and the class itsef is represented by the anchor image.

This property is captured in the equation for *Triplet Loss*

$$
loss = \frac{1}{N} \sum_{i=1}^N \left[||f(anc_i) - f(pos_i)||^2 + \alpha - ||f(anc_i) - f(neg_i)||^2 \right]_{+}
$$

Here, we make sure that there is a penalty as long as there are negative samples that are closer to the anchor than the margin value $\alpha$. The operator $[x]_+$ is equivalent to `max(0, x)`.


## Architecture

For this task, we will try to benefit of image recognition by loading model pretrained on ImageNet. Even though it tries to solve a different problem, we can benefit from the fact that the model learned how to interpret an image. Then we simply discard last layers that do the classification and add our new layers that will encode facial features.

![](https://software.intel.com/sites/default/files/managed/58/24/transfer-learning-fig2-schematic-depiction-of-transfer-learning.png)
*Transfer learning. Source: [Intel](https://software.intel.com/en-us/articles/use-transfer-learning-for-efficient-deep-learning-training-on-intel-xeon-processors)*

For our problem, we will import the weights of a CNN with architecture VGG19 pretrained on ImageNet as  a baseline. You are free to experiment with other models, e.g. Inception V3, ResNet and so on.

### Adding new layers

In order to adapt the model to facial recognition, add new layers to the model (baseline layers are in the Task section).

### Inference and Selecting Threshold

So far there we stated the existence of two thresholds: $T$ and $\alpha$. The first one is used to make the true class and the second (margin) to improve the construction of the face embedding space. These two values should be distinct because the margin $\alpha$ is merely a suggestion baked into the loss function, and there is no way to guarantee its strictness. It does not ensure that positive samples will fall into the hyper-sphere of radius $\alpha$ and all negative samples will be outside of it. Due to this, we need to come up with another threshold value to make a decision about the class of the current image, and this threshold will be $T$.

For inference, you need to compare two images and decide whether they are the same person or not. The decision should be made based on some threshold value. We selected the threshold of -0.8 based on the model performance after 500 epochs (assuming $R$ is defined the same way as in the section *Triplet Loss*). The actual threshold should be selected after the model was trained for hundreds of hours, the threshold is chosen with k-fold cross-validation or with a held-out dataset.

## Data Description

For training the model we are going to use preprocessed [Labeled Faces in the Wild](http://vis-www.cs.umass.edu/lfw/) dataset. All the faces were aligned and cropped. You can read more about this [here](https://hackernoon.com/building-a-facial-recognition-pipeline-with-deep-learning-in-tensorflow-66e7645015b8).

### Dataset Structure

The dataset structure is preserved and has the following form.
```
# Directory Structure
# ├── Tyra_Banks
# │ ├── Tyra_Banks_0001.jpg
# │ └── Tyra_Banks_0002.jpg
# ├── Tyron_Garner
# │ ├── Tyron_Garner_0001.jpg
# │ └── Tyron_Garner_0002.jpg
```
Where for every person there is a folder with the photos of faces. There are people with only one photo available: do not include them in the training process.

### Test Data ([Download](https://www.dropbox.com/s/loz4ijaxcor3l5l/test_set.csv?dl=1))
Test set is provided in the form of csv file

| Anchor | Positive | Negative |
| -------- | -------- | -------- |
| Vincent_Brooks/Vincent_Brooks_0002.jpg     | Vincent_Brooks/Vincent_Brooks_0006.jpg     | Gerhard_Schroeder/Gerhard_Schroeder_0007.jpg     |

We are going to evaluate the training based on classification accuracy and test set loss. The test set table contains 400 rows. This implies 800 pairs to compare and 400 decisions (same/different) to make. *Test set accuracy* is the accuracy score on those 400 comparisons.

# Task details

**First step:** Load the VGG19 pretrained model and discard last fully connected layers. You are free to experiment with other models, e.g Inception V3, ResNet and others.

**code for pytorch:**
```
model = models.vgg19(pretrained=True)
model = torch.nn.Sequential(*list(model.children())[:-1])
for param in model.parameters():
  param.requires_grad = False
```

**Second step:** Add new layers to the pretrained model. The baseline layers are the following:
- Dence layer with 512 units with sigmoid activation
- Dence layer with 256 units with sigmoid activation
- Dence layer with 128 units with tanh activation

*Advice: you can create another model containing new layers and connect it with VGG19 in forward method*

**Third step:** Specify triplet loss. It is recommended to create a separate class to calculate the triplet loss. Given an anchor, positive and negative images triplet loss can be computed as:
```
loss = mean((anchor - positive)**2) - mean((anchor - negative)**2) + margin
loss = min(loss, 0)
```

You may need to use following functions:
```
torch.mean() 
torch.clamp()
```

**Fourth step:** Load images. The following function might be helpful to load one image to memory, resize it in order to be feed to the network.
```
from PIL import Image

def read_and_resize(filename):
    img = Image.open(filename)
    transform = transforms.Compose([            
     transforms.Resize(299),                    
     transforms.ToTensor(),   
     transforms.Normalize(mean=(0.5,), std=(0.5,))
    ])
    img_t = transform(img)
    return torch.unsqueeze(img_t, 0)
```

**Fifth step:** Generating minibatches. The simplest policy for creating minibatches is the following:
1. During one epoch we will iterate over all anchors
2. For every anchor select one positive example. Given that some people have more than two images available, randomly sample the positive image. Make sure anchor and positive example do not match.
3. For every anchor sample a negative example. This could be any image of a different person.

**Sixth step:** Train the model. The baseline parameters for training are:
- Adam optimizer
- LR = 0.0001
- 500 epochs

You are free to change them.

You might need to use the following functions:
```
face_recognition_model.optimizer.zero_grad()
triplet_loss.backward()
face_recognition_model.optimizer.step()
```

**Seventh step:** Calculate the accuracy on a test set. Test set contains 800 pairs. For a given anchor you have to make a decision if it is a *positive person*, *negative person* or neither of them. based on similarity measure and threshold T. Then the accuracy will be the number of times when the anchor was correctly recognized.

| Anchor | Positive | Negative |
| -------- | -------- | -------- |
| Vincent_Brooks/Vincent_Brooks_0002.jpg     | Vincent_Brooks/Vincent_Brooks_0006.jpg     | Gerhard_Schroeder/Gerhard_Schroeder_0007.jpg     |

**Similarity measure can be written as:**
```
similarity(x,y) = - (l2_norm(model(x) - model(y)))**2
```

# Advanced topics
There are bunch of other loss functions that you can use instead of triplet loss:
- [ArcFace](https://www.google.com/search?q=arcface&rlz=1C5CHFA_enRU785RU785&oq=arcface&aqs=chrome.0.69i59j0l4j69i61.1479j0j7&sourceid=chrome&ie=UTF-8)
- [Focal Loss](https://github.com/opencv/openvino_training_extensions/blob/develop/pytorch_toolkit/face_recognition/losses/am_softmax.py#L36)
# References
- [FaceNet](https://arxiv.org/pdf/1503.03832.pdf)
- [Creating Frozen Graph From Checkpoints in Tensorflow](https://blog.metaflow.fr/tensorflow-how-to-freeze-a-model-and-serve-it-with-a-python-api-d4f3596b3adc)


## Other relevant resources

- https://www.superdatascience.com/opencv-face-detection/
- https://github.com/ageitgey/face_recognition
- https://github.com/deepinsight/insightface
- https://medium.freecodecamp.org/making-your-own-face-recognition-system-29a8e728107c
- https://kpzhang93.github.io/MTCNN_face_detection_alignment/index.html
- https://github.com/davidsandberg/facenet
- https://towardsdatascience.com/transfer-learning-in-tensorflow-9e4f7eae3bb4
- https://towardsdatascience.com/review-inception-v4-evolved-from-googlenet-merged-with-resnet-idea-image-classification-5e8c339d18bc
- https://medium.com/@RaghavPrabhu/cnn-architectures-lenet-alexnet-vgg-googlenet-and-resnet-7c81c017b848
- https://culurciello.github.io/tech/2016/06/20/training-enet.html
- https://towardsdatascience.com/lossless-triplet-loss-7e932f990b24
