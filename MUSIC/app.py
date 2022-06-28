import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 


from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras.utils.np_utils import to_categorical
from csv import writer
df = pd.read_csv("./data_moods.csv")


#Define the features and the target
col_features = df.columns[6:-3]
X= MinMaxScaler().fit_transform(df[col_features])
X2 = np.array(df[col_features])
Y = df['mood']


#Normalize the features

# X= MinMaxScaler().fit_transform(X)
# #Encode the labels (targets)
# encoder = LabelEncoder()
# encoder.fit(Y)
# encoded_y = encoder.transform(Y)
# #Split train and test data with a test size of 20%
# X_train,X_test,Y_train,Y_test = train_test_split(X,encoded_y,test_size=0.2,random_state=15)
# print(X_test)


#Encodethe categories
encoder = LabelEncoder()
encoder.fit(Y)
encoded_y = encoder.transform(Y)





X_train,X_test,Y_train,Y_test = train_test_split(X,encoded_y,test_size=0.2,random_state=15)

target = pd.DataFrame({'mood':df['mood'].tolist(),'encode':encoded_y}).drop_duplicates().sort_values(['encode'],ascending=True)
target





#Libraries to create the Multi-class Neural Network
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
#Import tensorflow and disable the v2 behavior and eager mode
import tensorflow as tf
from sklearn import model_selection   
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

# #Function that creates the structure of the Neural Network
def base_model():
    #Create the model
    model = Sequential()
#   Add 1 layer with 8 nodes,input of 4 dim with relu function
    model.add(Dense(8,input_dim=10,activation='relu'))
#   Add 1 layer with output 3 and softmax function
    model.add(Dense(4,activation='softmax'))
#Compile the model using logistic loss function and adam     optimizer, accuracy correspond to the metric displayed
    model.compile(loss='categorical_crossentropy',optimizer='adam',
              metrics=['accuracy'])
    return model
#Configure the estimator with 300 epochs and 200 batchs. the build_fn takes the function defined above.
estimator = KerasClassifier(build_fn=base_model,epochs=300,
                            batch_size=200)



# #Library to evaluate the model
# from sklearn.model_selection import cross_val_score, KFold


# #Evaluate the model using KFold cross validation
# kfold = KFold(n_splits=10,shuffle=True)
# results = cross_val_score(estimator,X,encoded_y,cv=kfold)
# #print("%.2f%% (%.2f%%)" % (results.mean()*100,results.std()*100))


# #Train the model with the train data
# estimator.fit(X_train,Y_train)
# y_preds = estimator.predict(X_test)
# from sklearn.metrics import accuracy_score
# print("Accuracy Score",accuracy_score(Y_test,y_preds)) 


# #Saving the  model to  use it later on
# # mus_json = estimator.to_json()
# # with open("mus.json", "w") as json_file:
# #     json_file.write(mus_json)
# # estimator.save_weights("mus.h5")
# # #Predict the model with the test data
# # # y_preds = estimator.predict(X_test)



# # # #Create the confusion matrix using test data and predictions
# # # cm = confusion_matrix(Y_test,y_preds)
# # # #plot the confusion matrix
# # # ax = plt.subplot()
# # # sns.heatmap(cm,annot=True,ax=ax)
# # # labels = df['mood'].tolist()
# # # ax.set_xlabel('Predicted labels')
# # # ax.set_ylabel('True labels')
# # # ax.set_title('Confusion Matrix')
# # # ax.xaxis.set_ticklabels(labels)
# # # ax.yaxis.set_ticklabels(labels)
# # # plt.show()
# # #Show the accuracy score 
# # print("Accuracy Score",accuracy_score(Y_test,y_preds))

# #Import the Script helpers.py
from sklearn.pipeline import Pipeline
from helpers import *
# def predict_mood(id_song):
#     #Join the model and the MinMaxScaler in a Pipeline
#     pip = Pipeline([('minmaxscaler',MinMaxScaler()),('keras',
#                      KerasClassifier(build_fn=base_model,epochs=300,
#                             batch_size=200))])
# #Fit the Pipeline
#     pip.fit(X,encoded_y)
    
#    #Obtain the features of the song (Function created on helpers.py)
#     preds = get_songs_features(id_song)
# #Pre-processing the input features for the Model
#     preds_features = np.array(preds[0][6:-2]).reshape(-1,1).T
   
#     #Predict the features of the song
#     results = pip.predict(preds_features)
#     mood = results[0]

    
#     #Obtain the name of the song and the artist
#     name_song = preds[0][0]
#     artist = preds[0][2]
# #Store the name,artist and mood of the song to print.
#     result_pred=print("{0} by {1} is a {2}  song".format(name_song,
#                                                  artist,mood))
#     return result_pred










# def base_model():
#     #Create the model
#     model = Sequential()
#     #Add 1 layer with 8 nodes,input of 4 dim with relu function
#     model.add(Dense(8,input_dim=10,activation='relu'))
#     #Add 1 layer with output 3 and softmax function
#     model.add(Dense(4,activation='softmax'))
#     #Compile the model using sigmoid loss function and adam optim
#     model.compile(loss='categorical_crossentropy',optimizer='adam',
#                  metrics=['accuracy'])
#     return model
# #Configure the model
# estimator = KerasClassifier(build_fn=base_model,epochs=300,batch_size=200,verbose=0)
# #Evaluate the model using KFold cross validation
# kfold = model_selection.KFold(n_splits=10,shuffle=True)
# results = cross_val_score(estimator,X,encoded_y,cv=kfold)
# #print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100,results.std()*100))

estimator.fit(X_train,Y_train)

y_preds = estimator.predict(X_test)
cm = confusion_matrix(Y_test,y_preds)
ax = plt.subplot()
sns.heatmap(cm,annot=True,ax=ax)

labels = target['mood']
ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')
ax.set_title('Confusion Matrix')
ax.xaxis.set_ticklabels(labels)
ax.yaxis.set_ticklabels(labels)
plt.show()

#print("Accuracy Score",accuracy_score(Y_test,y_preds))


def predict_mood(id_song):
    #Join the model and the scaler in a Pipeline
    pip = Pipeline([('minmaxscaler',MinMaxScaler()),('keras',KerasClassifier(build_fn=base_model,epochs=300,
                                                                             batch_size=200,verbose=0))])
    #Fit the Pipeline
    pip.fit(X2,encoded_y)

    #Obtain the features of the song
    preds = get_songs_features(id_song)
    #Pre-process the features to input the Model
    preds_features = np.array(preds[0][6:-2]).reshape(-1,1).T

    #Predict the features of the song
    results = pip.predict(preds_features)

    mood = np.array(target['mood'][target['encode']==int(results)])
    
    #Adding details of predicted song to csv file
    # Import writer class from csv module
    

    # List
    List=[preds[0][0],preds[0][1],preds[0][2],mood[0],preds[0][-2],preds[0][-1]]

    # Open our existing CSV file in append mode
    # Create a file object for this file
    with open('predicted_songs.csv', 'a') as f_object:

        # Pass this file object to csv.writer()
        # and get a writer object
        writer_object = writer(f_object)
        print(List)

        # Pass the list as an argument into
        # the writerow()
        writer_object.writerow(List)

        #Close the file object
        f_object.close()




    name_song = preds[0][0]
    artist = preds[0][2]

    return print("{0} by {1} is a {2} song".format(name_song,artist,mood[0].upper()))
    #print(f"{name_song} by {artist} is a {mood[0].upper()} song")
    





predict_mood('3oBiPP2S71AeRIxcj1aSdK')








