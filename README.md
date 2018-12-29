#
# Machine Learning Engineer Nanodegree

## **Capstone Project**

Ajay Singh
Dec 26th, 2018

##
# Definition

## **Project Overview**

### Creating machine learning based application which can capture frames from video and do analysis of that frames picture. So, what is Video about – almost everyone has internet on their home, and we use router/modem/gateways/cable boxes for getting internet connection/wifi/tv services. And we all get sometime these problems with our home devices and call customer care for that. So, my proposal is to create software which will capture a video of device (router/modem/gateways/cable boxes) and identify which lights are on/off and what color they have. Based on that data, we can decide what issue we have with that device like ethernet cable is loose or internet is not available and notify backend system to fix it or call customer care.

![Alt](prj.png?raw=true)

Something like below problem/solution –

[https://link.springer.com/chapter/10.1007/978-3-540-74853-3\_16](https://link.springer.com/chapter/10.1007/978-3-540-74853-3_16)

[https://pure.qub.ac.uk/portal/files/17844756/machine.pdf](https://pure.qub.ac.uk/portal/files/17844756/machine.pdf)

## **Problem Statement**

Home Router issues like ethernet cable loose, no internet connection, wifi off etc.  So customers call customer care which takes time and also some customer care support person to look and understand problem. Customers get frustrated most of the time with wait time/explanations and companies have to provide some support person on call even it is very small problem. With Machine learning we can achieve better User experience and save customer time and also save money/time for Company too.

Business Problem:

- Long wait times
- Unnecessary expensive dispatches
- Lack of digital Capabilities
- Unoptimized resource utilization

## **Metrics**

### As it is multi-classification problem, I am using loss metrics as **categorical\_crossentropy** , Optimizer as **rmsrop** and overall performance metrics for CNN model – **metrics.**

### Like example –

### model.compile(loss=&#39;categorical\_crossentropy&#39;, optimizer=&#39;rmsprop&#39;, metrics=[&#39;accuracy&#39;])

### **categorical\_crossentropy**

### keras.losses.categorical\_crossentropy(y\_true, y\_pred)

### **RMSprop**

### keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)

### **RMSProp optimizer**.

### It is recommended to leave the parameters of this optimizer at their default values (except the learning rate, which can be freely tuned). This optimizer is usually a good choice for recurrent neural networks.

### **Accuracy Score**

### Defined as % labels correctly classified when comparing model prediction vs actual. This metric will be computed on the training dataset and the validation set which is the population for which we have labels for the images. This is the secondary metric that will be computed and shared but the select of the optimal model will be based on minimizing log loss. It is expected that the model with the lowest log loss will also have one of the best accuracy score.

**My dataset looks like below:**

├── test

│   ├── PowerOn

│   ├── PowerOn\_2hz\_5hz

│   └── PowerOn\_Internet\_1\_2\_3\_4

├── train

│   ├── PowerOn

│   ├── PowerOn\_2hz\_5hz

│   └── PowerOn\_Internet\_1\_2\_3\_4

└── valid

│   ├── PowerOn

│   ├── PowerOn\_2hz\_5hz

│   └── PowerOn\_Internet\_1\_2\_3\_4


**Some Image Examples**** : -**

![Alt](Picture1.png?raw=true)

![Alt](Picture2.png?raw=true)

![Alt](Picture3.png?raw=true)


(above one with Tensorflow API)

## **Algorithms and Techniques**

Deep Learning is becoming a very popular subset of machine learning due to its high level of performance across many types of data. A great way to use deep learning to classify images is to build a convolutional neural network (CNN). The Keras library in Python makes it pretty simple to build a CNN.

![Alt](cnn.png?raw=true)

As I don&#39;t have much data and it is custom data (cannot find on internet, took from myself), I tried multiple approaches to find out results and tried to evaluate results.

**├──**  **bottle\_neck\_**** cnn****(****VGG16 Bottle-Neck approach with Image Augmentation)**

**│   ├── RouterLightClassifier.html**

**│   ├── RouterLightClassifier.ipynb**

**│   ├── bottleneck\_fc\_model.h5**

**│   ├── bottleneck\_features\_train.npy**

**│   ├── bottleneck\_features\_validation.npy**

**│   ├── class\_indices.npy**

**│   └── data**

**├──**  **cnn\_from\_scratch**** (****CNN approach with/without Image Augmentation)**

**│   ├── RouterLightClassifier-without-augmentation.ipynb**

**│   ├── RouterLightClassifier.ipynb**

**│   ├── data**

**│   ├── first\_try.h5**

**│   └── weights.best.from\_scratch1.hdf5**

**└──** **tf\_api\_code (****Using Tensorflow API and train model with label/images)**

**│   ├── data**

**│   ├── models**

**│   ├── nwd-tf**

**│   ├── object\_detection**

**│   ├── object\_detection\_tutorial.ipynb**

**│   └── test\_images**

**Implementation**

**Model construction** depends on machine learning algorithms. In this projects case, it was neural networks.

Such an algorithm looks like:

1. Begin with its object: model = Sequential() - Sequential is the easiest way to build a model in Keras. It allows you to build a model layer by layer.
2. then consist of layers with their types: model.add(_type\_of\_layer()_) - to add layers to our model.
3. after adding a sufficient number of layers the model is compiled. At this moment Keras communicates with TensorFlow for construction of the model. During model compilation it is important to write a loss function and an optimizer algorithm. It looks like: model.comile(loss= &#39;name\_of\_loss\_function&#39;, optimizer= &#39;name\_of\_opimazer\_alg&#39; ) The loss function shows the accuracy of each prediction made by the model.
4. **Max Pooling 2D** layer is pooling operation for spatial data. Numbers 2, 2 denote the pool size, which halves the input in both spatial dimensions.
5. Activation is the activation function for the layer. The activation function we will be using for layers is the ReLU, or Rectified Linear Activation. This activation function has been proven to work well in neural networks.
6. There is also &#39;Flatten&#39; layer. Flatten serves as a connection between the convolution and dense layers.
7. &#39;Dense&#39; is the layer type we will use in for our output layer. Dense is a standard layer type that is used in many cases for neural networks.
8. The activation is &#39;softmax&#39;. Softmax makes the output sum up to 1 so the output can be interpreted as probabilities. The model will then make its prediction based on which option has the highest probability.

Before model training it is important to scale data for their further use.

After model construction it is time for **model training.** In this phase, the model is trained using training data and expected output for this data.

It&#39;s look this way: model.fit(training\_data, expected\_output).

Progress is visible on the console when the script runs. At the end it will report the final accuracy of the model.

Once the model has been trained it is possible to carry out **model testing.** During this phase a second set of data is loaded. This data set has never been seen by the model and therefore it&#39;s true accuracy will be verified.

After the model training is complete, and it is understood that the model shows the right result, it can be saved by: model.save(&quot;name\_of\_file.h5&quot;).

Finally, the saved model can be used in the real world. The name of this phase is **model evaluation**. This means that the model can be used to evaluate new data.

**Image Augmentation**

Using little data is possible when the image is preprocessing with Keras ImageDataGenerator class. This class can create a number of random transformations, which helps to increase the number of images when it is needed.

**CNN From Scratch: -**

**Without**  **Image Augmentation:**

CNN Model: -


![Alt](cnn-1.png?raw=true)

![Alt](cnn1-g.png?raw=true)

**       **



**With Image Augmentation**** :**

CNN Model: -

![Alt](cnn2.png?raw=true)

![Alt](cnn2-a.png?raw=true)

![Alt](cnn2-g.png?raw=true)


**Bottle-Neck CNN From Scratch: -**

![Alt](cnn3.png?raw=true)

![Alt](cnn3-g.png?raw=true)

**Tensorflow API: -**

I experimented with training a custom object detector using TensorFlow&#39;s object detection API to detect three labels of router. The files for this project can be found in the under tf\_api\_code project folder. I followed the [Raccoon detector](https://towardsdatascience.com/how-to-train-your-own-object-detector-with-tensorflows-object-detector-api-bec72ecfe1d9) tutorial for this project.

Training a custom object detector takes place in mostly 3 parts:

1. Creating the data set
2. Training an object detector model on your data set
3. Testing the results of your custom object detector model

Here are the results from my training and evaluation jobs. In total, I ran it over about one hour/22k steps with a batch size of 24 but I already achieved good results in about 40mins.

This is how the total loss evolved:

![Alt](tens-g.png?raw=true)

##
# Conclusion

**Reflection**

Gather images as many for different status(lights) of device. Process images and create three sets – train, validate and test. Use CNN with VGG16 model and train model with train set of images. Validate with validate set and run test cases with test sets.  Measure accuracy of prediction and change CNN model designs and if necessary, try other benchmark models. Finalize model and store for future use.

Understanding of deep learning, convolution neural networks and Tensorflow through courses and books. When I started I had little to no understanding of this field. This was the pre-phase to give me the basic tools to get started.

Pre-Processing of the dataset – Learning about image processing was the next phase. Deciding what pre-processing must be done with the first iteration (resize image, flatten to rgb, converting to array data) vs what can be tried later after I get comfortable (image augmentation, colored images)

The depth of neural network materially affects accuracy. As the network gets deeper, its ability to learn finer features improves. This is intuitive. However, I found that increasing the number of neurons in each layer didn&#39;t always have a positive effect on accuracy and sometimes decreased accuracy. The best setting for me was a deep network with small number of neurons in each layer

**Improvement**

I believe that if I collect more devices light combination images and increase image data size, I can achieve very good CNN model which we can rely to accurately find which light combination are ON on network device. I also think that collecting different LED lights separately like 1,2,3,4, internet, power etc. and then train separately and use RNN or Tensorflow API, we can achieve good results.

Use of other more modern techniques that can help reduce overfitting, Better input pre-processing to help the model **,** Use of external data (In my case, I did not find any)