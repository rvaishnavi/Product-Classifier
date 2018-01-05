# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 12:29:47 2017

@author: vaish
"""
import pandas as pd
import numpy as np


# Importing the dataset
#dataset = pd.read_excel('Training_Data_Assessment.xlsx', sheet_name=0).iloc[:200]
dataset = pd.read_excel('Training_Data_Assessment.xlsx', sheet_name=0)
 
# Importing pretrained model for image feature extraction
from keras import applications
model = applications.resnet50.ResNet50(weights='imagenet', include_top=False, pooling='avg')

# Dateframe to store image features
imgf = pd.DataFrame()

# Get all the features of images from the Urls provided
import urllib.request
from keras.applications.resnet50 import preprocess_input
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img

for index, value in dataset['ImageUrl'].items():
    print(index)
    print(value)
    
    # Save the image from url to a file
    urllib.request.urlretrieve(value,"tempimage.jpg")
    img_path = 'tempimage.jpg'
    
    # load image setting the image size to 224 x 224
    img = load_img(img_path, target_size=(224, 224))
    
    # Convert image to numpy array and process
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    
    # extract the features
    features = model.predict(x)[0]
    print(features)
    
    # convert the features to a row matrix and save in a data frame
    data = pd.DataFrame(np.matrix(features))
    imgf = imgf.append(data, ignore_index=True)


#print(imgf)
print(imgf.shape)
print(dataset.shape)

# CategoryName, BrandName, Title and the (2048) features extracted from the image are saved in a dataframe
dataset = pd.concat([dataset.iloc[:,0:4],imgf.iloc[:]],axis =1)
print(dataset.iloc[:,5:])
dataset = dataset.sample(frac=1)

# features and target
X = dataset.iloc[:, 2:]
Y = dataset['CategoryName']
print(X)

# test and train data split
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, random_state = 0)

# verify model inputs shape
print(X.shape)
print(Y.shape)
print (X_train.shape)
print(Y_train.shape)

# Dataframe mapper - to Label to Brand Names and Vectorize the Titles
from sklearn.feature_extraction.text import CountVectorizer
from sklearn_pandas import DataFrameMapper
import sklearn.preprocessing
mapper = DataFrameMapper([('BrandName',sklearn.preprocessing.LabelBinarizer()),\
                          ('Title',CountVectorizer(stop_words={'English'}))
                          ])
mapper2 = DataFrameMapper([('CategoryName',None)])


# pipeline the training
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
pipeline = Pipeline([
        ('feature_mapper', mapper),
        ('classifier', LogisticRegression())
    ])
    

# verify cross val score    
from sklearn.model_selection import StratifiedKFold, cross_val_score

skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
scores = cross_val_score(pipeline, X_train, Y_train, cv=skf)
print('Accuracy estimates: {}'.format(scores))


#Fitting logistic regression to the training set
classifier = pipeline.fit(X=X_train,y = Y_train)

# predict for the test set
print (X_test)
Y_pred = classifier.predict(X_test)

# verify test mean
print("Test mean: ", np.mean(Y_pred == Y_test))

#########################################################################

# export the model to a file
from sklearn.externals import joblib
# save the model to disk
filename = 'finalized_modeltxt.txt'
joblib.dump(classifier, filename)

#########################################################################

# load the model from disk to check a sample
loaded_model = joblib.load(filename)
result = loaded_model.score(X_test, Y_test)
print(result)

#Sample Validation
# random sample data to test
vdata = pd.DataFrame({'BrandName':['1byone','3ware','Cisco'],\
                     'Title':['1byone 5GHz Wireless HDMI Streaming Media Player, WiFi Display Dongle Share Videos, Images, Docs, Live Camera and Music from All Smart Devices to TV, Monitor or Projector',\
                              '8-PORT Int., 6GB/S Sata+sas, Pcie 2.0, 512MB; In The Box: 3WARE Sas 9750-8I, Qig',\
                              'Cisco AIR-CAP2602I-B-K9 Aironet 2600 Series Ap Networking Device'],\
                     'ImageUrl':['https://images-na.ssl-images-amazon.com/images/I/41-QFubslLL.jpg',\
                                          'http://ecx.images-amazon.com/images/I/21Yxo94fmrL.jpg',\
                                          'http://ecx.images-amazon.com/images/I/118SltpIKLL.jpg']})

imgfx = pd.DataFrame()
for index, value in vdata['ImageUrl'].items():
    print(value)
    urllib.request.urlretrieve(value,"tempimage.jpg")
    img_path = 'tempimage.jpg'
    img = load_img(img_path, target_size=(224, 224))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    # extract the features
    features = model.predict(x)[0]
    data = pd.DataFrame(np.matrix(features))
    imgfx = imgfx.append(data, ignore_index=True)
    
#print(vdata)
tdata = pd.concat([vdata.iloc[:,[0,2]],imgfx.iloc[:]],axis =1)
print(tdata)

#cat = classifier.predict(tdata)
category = loaded_model.predict(tdata)

# Verify the category
print(category)
######################################
