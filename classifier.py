import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np


data_dict = pickle.load(open('./data.pickle', 'rb')) #load the dataset

data = np.asarray(data_dict['data']) #read the dataset into numpy arrays
labels = np.asarray(data_dict['labels'])

#divide the dataset into two sections such that 80% given to train and 20% given to test the model
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels) 

model = RandomForestClassifier() #I used randomforest classifier here because i have categorical data

model.fit(x_train, y_train) #train the model with 80% of data

y_predict = model.predict(x_test) #test the model with remaining 20% of data

score = accuracy_score(y_predict, y_test) #check the score

print('percentage of samples classified correctly =',score * 100)

f = open('model.p', 'wb')  #Opens a file named model.p in binary write mode inorder to store the trained model
pickle.dump({'model': model}, f) #save the trained model as a pickle file
f.close()