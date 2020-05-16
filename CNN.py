import numpy as np 
import keras
import random
import cv2
from keras.layers import Conv2D, MaxPooling2D, Activation, Flatten, Dense

C = 2
N = 800
epochs = 1600

TrainSet = []
TestSet = []
LabelTrain = []
LabelTest = []

pathTrain = 'C:\\Users\ADMIN\Desktop\CNN\Train\\'
pathTest = 'C:\\Users\ADMIN\Desktop\CNN\Test\\'

#Load Train Set
print('Loading Train Set!!!')
ListTrain = [_ for _ in range(1, N + 1)]
random.shuffle(ListTrain)
for i in ListTrain:
    path = pathTrain
    if i <= int(N / 2):
        path = path + 'Hanquoc\\' + 'S- ' + str(i) + '.jpg'
        LabelTrain.append([1, 0])
    else:
        path = path + 'Ngoclinh\\' + 'S- ' + str(i - int(N / 2)) + '.jpg'
        LabelTrain.append([0, 1])
    img = cv2.imread(path, 1)
    img = cv2.resize(img, (96, 96))
    TrainSet.append(img)

LabelTrain = np.array(LabelTrain)
TrainSet = np.array(TrainSet)
TrainSet = TrainSet / 255
print('Number element of Train Set: ', len(TrainSet))

#Load Test Set
print('Loading Test Set!!!')
ListTest = [_ for _ in range(1, int(N / 4) + 1)]
random.shuffle(ListTest)
for i in ListTest:
    path = pathTest
    if i <= int(N / 8):
        path = path + 'Hanquoc\\' + 'S-1 (' + str(i) + ').jpg'
        LabelTest.append([1, 0])
    else:
        path = path + 'Ngoclinh\\' + 'S- 1 (' + str(i - int(N / 8)) + ').jpg'
        LabelTest.append([0, 1])
    img = cv2.imread(path, 1)
    img = cv2.resize(img, (96, 96))
    TestSet.append(img)

LabelTest = np.array(LabelTest)
TestSet = np.array(TestSet)
TestSet = TestSet / 255
print('Number element of Test Set: ', len(TestSet))

#Building model
model = keras.Sequential()

# First Conv
model.add(Conv2D(8, 3, padding= 'same', input_shape= TrainSet.shape[1:])) # Add Conv layer
model.add(Activation('relu')) # Add Activation Function
model.add(Conv2D(8, 3, padding= 'same')) # Add Conv layer
model.add(Activation('relu')) #Add Activation Function
model.add(MaxPooling2D(pool_size= (2, 2))) # Add Pooling layer

# Second Conv
model.add(Conv2D(16, 3, padding= 'same')) # Add Conv layer
model.add(Activation('relu')) # Add Activation Function
model.add(Conv2D(16, 3, padding= 'same')) # Add Conv layer
model.add(Activation('relu')) # Add Activation Function
model.add(MaxPooling2D(pool_size= (2, 2))) # Add Pooling layer

# Third Conv
model.add(Conv2D(32, 3, padding= 'same')) # Add Conv layer
model.add(Activation('relu')) # Add Activation Function
model.add(Conv2D(32, 3, padding= 'same')) # Add Conv layer
model.add(Activation('relu')) # Add Activation Function
model.add(MaxPooling2D(pool_size= (2, 2))) # Add Pooling layer

# Full Conected
model.add(Flatten()) # Flatten
model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dense(2))
model.add(Activation('softmax'))

#Train Model
model.compile(loss= 'categorical_crossentropy', optimizer= keras.optimizers.SGD(learning_rate= 0.01), metrics= ['accuracy'])
model.fit(TrainSet, LabelTrain, batch_size= N, epochs= epochs, validation_data= (TestSet, LabelTest))

#Get Score
score = model.evaluate(TestSet, LabelTest)

print('Test Loss: ', score[0])
print('Test accuracy', score[1])