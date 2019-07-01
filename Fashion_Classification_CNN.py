import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dropout,Dense,Flatten
from keras.layers.convolutional import Conv2D,MaxPooling2D
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from keras.layers.normalization import BatchNormalization
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from keras import optimizers

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

#load the data
training_set = pd.read_csv('fashion-mnist_train.csv')
testing_set = pd.read_csv('fashion-mnist_test.csv')

#Preprocess the data 
# Reshape the given csv into the original images


# Creation of validation dataset to improve model performance evalutation
X = np.array(training_set.iloc[:,1:])
X = X.reshape(X.shape[0],28,28,1).astype('float32')
X = X/255
y = to_categorical(np.array(training_set.iloc[:,0])) 
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.33, random_state=seed)

X_test = np.array(testing_set.iloc[:,1:])
X_test = X_test.reshape(X_test.shape[0],28,28,1).astype('float32')
X_test = X_test/255
y_test = to_categorical(np.array(testing_set.iloc[:,0]))

# Check if the reshaping worked by displaying the 9 first images 

for i in range(0,9):
    plt.subplot(330 + 1 + i)
    plt.imshow(np.squeeze(X_train[i]), cmap='gray')
plt.show()

# Building a cnn model
shape = (28,28,1)
number_of_classes = y_test.shape[1]
def create_model():
    model = Sequential()
    model.add(Conv2D(32,(3,3),activation ='relu',input_shape = shape ))
    model.add(Conv2D(32,(3,3),activation ='relu',input_shape = shape ))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(64,(3,3),activation ='relu' ))
    model.add(Conv2D(64,(3,3),activation ='relu' ))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(number_of_classes, activation='softmax'))
    model.compile(optimizer= 'adam' ,loss='categorical_crossentropy',metrics=['accuracy'])
    return model


# Early stopping to save the best model 
callbacks = [EarlyStopping(monitor='val_loss', patience=5, verbose=1),ModelCheckpoint('best_cnn.h5', monitor='val_loss', save_best_only=True)]

# Training the model
batch_size = 150
epochs = 20
model = create_model()
history = model.fit(X_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    callbacks = callbacks,
                    verbose=1,
validation_data=(X_val, y_val))

# Evaluate the best model

best_model = load_model('best_cnn2.h5')
score = best_model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# Make predictions about the classes
predictions = best_model.predict_classes(X_test,batch_size=150)  


# Save the classification

filename = 'best_cnn_classification.csv';
PictureID = list(range(1,len(predictions)+1))
submission = pd.DataFrame({'Clothe_category':predictions })
submission.to_csv(filename,header = True, index = False)
print('Saved file: ' + filename)

# Data Model visualization
print(best_model.summary())

# Studying misclassfications

target_names = ["Class {}".format(i) for i in range(number_of_classes)]
y_true = testing_set.iloc[:,0]

print(classification_report(y_true, predictions, target_names=target_names))


print(confusion_matrix(y_true, predictions))

#Let's display actual shirts
y_true = np.array(y_true)
shirt = predictions == 6
acurate_prediction = predictions == y_true 
real_shirt = np.nonzero( shirt & acurate_prediction)[0] 
for i, real_shirt in enumerate(real_shirt[:9]):
    plt.subplot(330 + 1 + i)
    plt.imshow(np.squeeze(X_test[real_shirt]), cmap='gray')
plt.show()

# Let's display misclassified shirts
y_true = np.array(y_true)
shirt = predictions == 6
prediction_error = predictions!=y_true 
fake_shirt = np.nonzero( shirt & prediction_error)[0] 

for i, fake_shirt in enumerate(fake_shirt[:9]):
    #plt.subplot(440 + 1 + i)
    plt.subplot(3,3,i+1)
    #plt.subplot(10,5,i+1)
    plt.imshow(np.squeeze(X_test[fake_shirt]), cmap='gray')
    #plt.title("Predicted class {}, Actual Class {}".format(predictions[fake_shirt], y_true[fake_shirt]))
    plt.title(y_true[fake_shirt])
    
#figure = plt.gcf()
plt.tight_layout()
plt.show()
#figure.savefig('misclassification.png')    
    


