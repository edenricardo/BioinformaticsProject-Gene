import numpy as np
from sklearn.preprocessing import label_binarize
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.linear_model import LogisticRegression


def load_data():

    train = np.load('train.npz')
    data = train['x']
    label = train['y']
    # label = label_binarize(label, classes=[0,1,2,3,4,5])
    print (data.shape, label.shape)

    return data, label

def loda_valid():

    valid = np.load('valid.npz')
    x_v = valid['x']
    y_v = valid['y']
    print (x_v.shape, y_v.shape)

    return x_v, y_v

def main():

    data, label = load_data()
    x_v, y_v = loda_valid()

    # PCA
    pca = PCA(n_components=500) # test
    pca.fit(data)
    data_pca=pca.transform(data)
    x_v_pca=pca.transform(x_v)
    print ("PCA;", data_pca.shape)

    # SVM
    clf = svm.LinearSVC(penalty='l1', dual=False)
    clf.fit(data_pca, label)

    print ("SVM:")
    pred = clf.predict(x_v_pca)  
    count = 0
    for i in range(590):
        prediction = pred[i]
        if prediction == y_v[i]:
            count += 1
        else:
            print (i, y_v[i], prediction)
    print ("Accuracy:", count, "/ 590 = ", count/590)


    # LR
    lr_clf = LogisticRegression(penalty='l1')
    lr_clf.fit(data_pca, label)

    print ("LR:")
    lr_pred = lr_clf.predict(x_v_pca)
    count = 0
    for i in range(590):
        prediction = lr_pred[i]
        if prediction == y_v[i]:
            count += 1
        else:
            print (i, y_v[i], prediction)
    print ("Accuracy:", count, "/ 590 = ", count/590)
    

if __name__ == '__main__':

    # load_data()
    main()
