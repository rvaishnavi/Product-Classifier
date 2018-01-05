# -*- coding: utf-8 -*-
"""
Created on Sat Dec 30 21:46:47 2017

@author: vaish
"""
import pandas as pd
import numpy as np
import urllib.request
from keras.applications.resnet50 import preprocess_input
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img

# Importing the test dataset
dataset = pd.read_excel('Data To Classify_Assessment (1).xlsx', sheet_name=0).iloc[0:100]

#dataset = pd.read_excel('Data To Classify_Assessment.xlsx', sheet_name=0).iloc[25:35]
#dataset = pd.read_excel('Data To Classify_Assessment.xlsx', sheet_name=0, header = 0,skiprows = 25, skip_footer = 6000)

# Only top 100 rows from the data to classify file is read to a datafraame (as few brand names are NULL in the test sheet)

# index staaring from 0
dataset = dataset.reset_index(drop = True)

# Importing pretrained model for image feature extraction
from keras import applications
model = applications.resnet50.ResNet50(weights='imagenet', include_top=False, pooling='avg')

imgfx = pd.DataFrame()

for index, value in dataset['ImageUrl'].items():
    print(index,value)
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
tdata = pd.concat([dataset.iloc[:,[1,2]],imgfx.iloc[:]],axis =1)
print(tdata)

# load the model from disk to check a sample
from sklearn.externals import joblib
filename = 'finalized_modeltxt.txt'
loaded_model = joblib.load(filename)
category = loaded_model.predict(tdata)

# Verify the category
print(category)

# store the predicted category in a dataframe
df = pd.DataFrame(columns = ['PredictedCategories'])
#df = pd.DataFrame(category)
df['PredictedCategories'] = category
print(df)


# write the predicted product categories into the existing xlsx file
from openpyxl import load_workbook

book = load_workbook('Data To Classify_Assessment (1).xlsx')
writer = pd.ExcelWriter('Data To Classify_Assessment (1).xlsx', engine='openpyxl') 
writer.book = book
writer.sheets = dict((ws.title, ws) for ws in book.worksheets)

df['PredictedCategories'].to_excel(writer,sheet_name='sample_data', startcol = 4, index = False)

writer.save()


