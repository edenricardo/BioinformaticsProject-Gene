import numpy as np
import keras
from keras import models
from keras import layers
from keras import regularizers
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split

def load_data():

    '''
    data = np.load('microarray.original.npy')
    data = np.transpose(data, (1,0))
    # print (data.shape)
    label = np.load('label.npy')
    # print (label.shape)

    # valid 
    x_data, x_valid, y_label, y_valid = train_test_split(data, label, test_size=0.1)
    print (x_data.shape)
    print (x_valid.shape)
    np.savez('train.npz', x=x_data, y=y_label)
    np.savez('valid.npz', x=x_valid, y=y_valid)
    '''
    
    train = np.load('train.npz')
    data = train['x']
    print (data.shape)
    label = train['y']
    label = keras.utils.to_categorical(label, 6)
    print (label.shape)

    return data, label


def main():

    data, label = load_data()
    x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.2)
    print(x_train.shape,y_train.shape)

    model = models.Sequential()
    model.add(layers.Dense(16,input_dim=22283,activation='relu',kernel_regularizer=regularizers.l2(0.01)))
    model.add(layers.Dense(16,activation='relu',kernel_regularizer=regularizers.l1(0.01)))
    model.add(layers.Dense(8,activation='relu',kernel_regularizer=regularizers.l1(0.01)))
    model.add(layers.Dense(6,activation='softmax'))

    model.summary()

    model.compile(optimizer=Adam(lr=0.0003),loss='categorical_crossentropy',metrics=['accuracy'])
    # test 'rmsprop'

    print ("Train......")
    history = model.fit(x_train,
                    y_train,
                    epochs=100,  # test
                    batch_size=16,
                    verbose = 1, 
                    validation_data=(x_test, y_test))

    print ("Evaluate......")
    score, accuracy = model.evaluate(x_test, y_test, batch_size=16)
    print('Score:', score, 'Accuracy:', accuracy)

    model.save('gene-dnn-reg.h5')


def pred_data():

    valid = np.load('valid.npz')
    x_v = valid['x']
    y_v = valid['y']
    # y_v = keras.utils.to_categorical(y_v, num_classes)

    model = models.load_model('gene-dnn-reg.h5')

    model.compile(optimizer=Adam(lr=0.0003),loss='categorical_crossentropy',metrics=['accuracy'])
    # test 'rmsprop'

    pred = model.predict(x_v)

    count = 0
    for i in range(590):
        prediction = np.argmax(pred[i])
        if prediction == y_v[i]:
            count += 1
        else:
            print (i, y_v[i], prediction)
    print ("Accuracy:", count, "/ 590 = ", count/590)
        


if __name__ == '__main__':

    # load_data()
    # main()
    pred_data()
