#!/usr/bin/env python
# coding: utf-8

# In[36]:


import tensorflow as tf


# In[37]:


from tensorflow import keras


# In[38]:


from tensorflow.keras.optimizers import RMSprop


# In[39]:


import numpy as np


# In[40]:


import pandas as pd


# In[41]:


from tensorflow.keras.preprocessing import image


# In[42]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator


# In[43]:


import matplotlib.pyplot as plt


# In[44]:


import os


# In[45]:


import cv2


# In[46]:


img=image.load_img ("C://Users//Admin//Desktop//Aishwarya Rai Images//training//good images//ash 11.JFIF")


# In[47]:


plt.imshow(img)


# In[48]:


cv2.imread ("C://Users//Admin//Desktop//Aishwarya Rai Images//training//good images//ash 11.JFIF")


# In[49]:


cv2.imread ("C://Users//Admin//Desktop//Aishwarya Rai Images//training//good images//ash 11.JFIF").shape


# In[50]:


train=ImageDataGenerator(rescale=1/206)
validation=ImageDataGenerator(rescale=1/206)


# In[51]:


train_dataset=train.flow_from_directory("C://Users//Admin//Desktop//Aishwarya Rai Images//training//",target_size=(150,150),batch_size=2,class_mode='binary')


# In[52]:


validation_dataset=validation.flow_from_directory("C://Users//Admin//Desktop//Aishwarya Rai Images//validation//",target_size=(150,150),batch_size=2,class_mode='binary')


# In[53]:


train_dataset.class_indices


# In[54]:


train_dataset.classes


# In[55]:


model=tf.keras.models.Sequential([tf.keras.layers.Conv2D(16,(3,3),activation ='relu',input_shape=(150,150,3)),
                                  tf.keras.layers.MaxPool2D(2,2),
#
                                  tf.keras.layers.Conv2D(32,(3,3),activation ='relu'),
                                  tf.keras.layers.MaxPool2D(2,2), 
#
                                  tf.keras.layers.Conv2D(64,(3,3),activation ='relu'),
                                  tf.keras.layers.MaxPool2D(2,2), 
## 
                                  tf.keras.layers.Flatten(),
##
                                  tf.keras.layers.Dense(512,activation ='relu'),
##
                                  tf.keras.layers.Dense(1,activation ='sigmoid')])


# In[56]:


model.compile(loss ='binary_crossentropy',
              optimizer= RMSprop(lr=0.001),
              metrics=['accuracy'])


# In[57]:


model_fit=model.fit(train_dataset,
                    steps_per_epoch=1,
                    epochs=30,
                   validation_data=validation_dataset)


# In[58]:


validation_dataset.class_indices


# In[59]:


dir_path=("C://Users//Admin//Desktop//Aishwarya Rai Images//training//good images//")
for i in os.listdir(dir_path):
    img=image.load_img(dir_path+'//'+i)
    plt.imshow(img)
    plt.show()
    
    


# In[60]:


dir_path=("C://Users//Admin//Desktop//Aishwarya Rai Images//training//bad images//")
for i in os.listdir(dir_path):
    img=image.load_img(dir_path+'//'+i,target_size=(150,150))
    plt.imshow(img)
    plt.show()


   
  


# In[61]:


dir_path=("C://Users//Admin//Desktop//Aishwarya Rai Images//training//good images//")
for i in os.listdir(dir_path):
    img=image.load_img(dir_path+'//'+i,target_size=(150,150))
    plt.imshow(img)
    plt.show()
    X = image.img_to_array(img)
    X = np.expand_dims(X,axis = 0)
    images= np.vstack([X])
    val = model.predict(images)
if val == 0:
    print ("images is bad")
else:
    print ("images is good")

   


# In[62]:


dir_path=("C://Users//Admin//Desktop//Aishwarya Rai Images//training//bad images//")
for i in os.listdir(dir_path):
    img=image.load_img(dir_path+'//'+i,target_size=(150,150))
    plt.imshow(img)
    plt.show()
    X = image.img_to_array(img)
    X = np.expand_dims(X,axis =0)
    images=np.vstack([X])
    val = model.predict(images)
if val == 0:
    print ("images is bad")
else:
    print ("images is good")


# In[ ]:




