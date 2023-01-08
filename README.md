### [Go to Projects Page](https://github.com/celik-muhammed/15P-Deep-Learning-Projects-with-Python/blob/master/README.md)

## DL_CNN Cats vs Dogs Classification Project
- The Dogs vs. Cats dataset is a common computer vision dataset in which pictures are classified as either including a dog or a cat.
- Train set includes 12500 cat-5026 dog images, validation set includes 1219 cat-1071 dog images and test set includes 6897 cat and dogs images together.
>- Data Sources: [Kaggle Cats and Dogs Dataset](https://www.kaggle.com/c/dogs-vs-cats/data), &nbsp;&nbsp; [microsoft kagglecatsanddogs_5340.zip](https://www.microsoft.com/en-us/download/details.aspx?id=54765)   
1. [DL_CNN Dogs vs Cat Classification Project Colab](./01-Cats-vs-Dogs-ImageDataGenerator/CNN_Project_Image_Classification_with_CNN_(catdogclassifier)_Student.ipynb)

## EXAMINED ADVANCED DATA AUGMENTATION TECHNIQUES

#### [CPU, GPU] GENERATOR: INGEST DATA and EXPLORE DATA with tf.keras.preprocessing.image.ImageDataGenerator(...)
2. [DL_CNN Dogs vs Cat Classification Project Kaggle-Colab](./01-Cats-vs-Dogs-ImageDataGenerator/README.md)
> <sub>https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator</sub> 

#### [CPU, GPU] FROM_DIRECTORY: INGEST DATA and EXPLORE DATA with tf.keras.utils.image_dataset_from_directory(...)
2. [DL_CNN Dogs vs Cat Classification Project Kaggle-Colab](./02-Cats-vs-Dogs-image_dataset_from_directory/README.md)
> <sub>https://www.tensorflow.org/api_docs/python/tf/keras/utils/image_dataset_from_directory</sub>
> <sub>[tf.data: Build TensorFlow input pipelines](https://www.tensorflow.org/guide/data)</sub>

#### [CPU, GPU, TPU] TFRecords: INGEST DATA and EXPLORE DATA with SAVE tf.io.TFRecordWriter(...)
2. [DL_CNN Dogs vs Cat Classification Project Kaggle-Colab](./03-Cats-vs-Dogs-TFRecordWriter/README.md)
> <sub>https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset</sub>
> <sub>https://www.tensorflow.org/api_docs/python/tf/io/TFRecordWriter</sub>

- #### [Sample Images - EXPLORE DATA)](README.md#Dataset-Sample-Images)

<div align='center'>
    
## Visually Compare Models Performance In a Graph    
<h3>Scores</h3>
<img src='https://i.ibb.co/k0Ncjh3/download.png' alt='' width=45%, height=300> 
<img src='https://i.ibb.co/SVSZ1kL/download.png' alt='' width=45%, height=300>    
<h3>Dataset Sample Images</h3>
<br>    
<img src='https://i.ibb.co/JFGXbdH/download.png' alt='' width=80%, height=400>
</div>

- [cats_vs_dogs](https://www.tensorflow.org/datasets/catalog/cats_vs_dogs)
- [The Keras Blog alternative](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html)
- [tf.keras.preprocessing.image.ImageDataGenerator](https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator)
- [Build TensorFlow input pipelines](https://www.tensorflow.org/guide/data)
- [Apply image augmentations to the data](https://www.tensorflow.org/hub/tutorials/cropnet_on_device)

    - [Convolutional Neural Network (CNN)](https://www.tensorflow.org/tutorials/images/cnn)
    - [Image classification](https://www.tensorflow.org/tutorials/images/classification)
    - [Keras: Computer Vision](https://keras.io/examples/vision/)
    - [data_augmentation.ipynb](https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/images/data_augmentation.ipynb#scrollTo=pkTRazeVRwDe)
  
 - [Classification: Accuracy](https://developers.google.com/machine-learning/crash-course/classification/accuracy#:~:text=Accuracy%20is%20one%20metric%20for,predictions%20Total%20number%20of%20predictions)
 - [F1 score calculation](https://hasty.ai/docs/mp-wiki/metrics/f-beta-score)
