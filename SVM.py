import numpy as np
import random
from sklearn.metrics import f1_score
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt





class Linear_Kernel():

    def forward(self,alphas,Labels,X,b,input):
        """
        :param alphas: array of alphas
        :param Labels: labels
        :param X:  array of X
        :param input: input vector
        :param b:  bias
        :return: scalr
        """
        fxi = float(np.multiply(alphas, Labels).T * (X * input.T)) + b
        return fxi

    def inner(self,vectorA,vectorB):
        return vectorA * vectorB.T


class Polynomial_Kernel():
    def __init__(self,P=3):
        self.P = P

    def forward(self,alphas,Labels,X,b,input):
        """
        :param alphas: array of alphas
        :param Labels: labels
        :param X:  array of X
        :param input: input vector
        :param b:  bias
        :return: scalr
        """
        fxi = float(np.multiply(alphas, Labels).T * np.power(X * input.T+1,self.P)) + b
        return fxi

    def inner(self,vectorA,vectorB):
        return np.power(vectorA * vectorB.T+1,self.P)


class RBF_Kernel():

    def __init__(self,sigma=3):
        self.sigma = sigma
    def forward(self,alphas,Labels,X,b,input):
        """
        :param alphas: array of alphas
        :param Labels: labels
        :param X:  array of X
        :param input: input vector
        :param b:  bias
        :return: scalr
        """
        aa = -np.multiply((input - X),(input - X))
        out = np.multiply(alphas, Labels).T* np.exp(np.sum(aa, axis=-1) / 2 * self.sigma * self.sigma)
        out = out.sum()+b
        return out

    def inner(self,vectorA,vectorB):
        out = np.exp(np.sum(-np.multiply((vectorA - vectorB),(vectorA - vectorB)), axis=-1) / 2 * self.sigma * self.sigma)
        return out.sum()


class SVM():
    def __init__(self,toler,maxIter,C,kernel):

        self.toler = toler
        self.maxIter = maxIter
        self.C = C
        self.alpha = None
        self.b = 0
        self.kernel = kernel

        self.print_interval = 500

    def train(self,data,label):
        """
        Simple SMO algorithm
        :return:
        """
        print("start training!!")
        dataMatrix = np.mat(data)
        labelMatrix = np.mat(label).transpose()
        b = 0.0
        iter = 0
        m, n = np.shape(dataMatrix)
        alpha = np.mat(np.zeros((m, 1)))
        print_count = 0
        iter_count = 0
        while iter < self.maxIter:
            alphapairChanged = 0
            for i in range(m):
                fxi = self.kernel.forward(alpha, labelMatrix, dataMatrix, b, dataMatrix[i, :])

                Ei = fxi - float(labelMatrix[i])
                if labelMatrix[i] * Ei < -self.toler and alpha[i] < self.C or labelMatrix[i] * Ei > self.toler and alpha[i] > 0:
                    j = self.selectJrand(i, m)
                    fxj = self.kernel.forward(alpha, labelMatrix, dataMatrix, b, dataMatrix[j, :])

                    Ej = fxj - float(labelMatrix[j])
                    alphaIOld = alpha[i].copy()
                    alphaJOld = alpha[j].copy()
                    if labelMatrix[i] != labelMatrix[j]:
                        L = max(0, alpha[j] - alpha[i])
                        H = min(self.C, self.C + alpha[j] - alpha[i])
                    else:
                        L = max(0, alpha[i] + alpha[j] - self.C)
                        H = min(self.C, alpha[j] + alpha[i])
                    if L == H:
                        continue
                    eta = 2.0 * self.kernel.inner(dataMatrix[i, :], dataMatrix[j, :]) - \
                          self.kernel.inner(dataMatrix[i, :],dataMatrix[i,:]) - self.kernel.inner(dataMatrix[j, :], dataMatrix[j, :])

                    if eta >= 0:
                        continue
                    alpha[j] -= labelMatrix[j] * (Ei - Ej) / eta
                    alpha[j] = self.clipAlpha(alpha[j], H, L)
                    if abs(alpha[j] - alphaJOld) < 0.00001:
                        continue
                    alpha[i] += labelMatrix[j] * labelMatrix[i] * (alphaJOld - alpha[j])
                    b1 = b - Ei - labelMatrix[i] * (alpha[i] - alphaIOld) * self.kernel.inner(dataMatrix[i, :],
                                                                                         dataMatrix[i, :]) \
                         - labelMatrix[j] * (alpha[j] - alphaJOld) * self.kernel.inner(dataMatrix[i, :], dataMatrix[j, :])
                    b2 = b - Ej - labelMatrix[i] * (alpha[i] - alphaIOld) * self.kernel.inner(dataMatrix[i, :],
                                                                                         dataMatrix[j, :]) \
                         - labelMatrix[j] * (alpha[j] - alphaJOld) * self.kernel.inner(dataMatrix[j, :], dataMatrix[j, :])

                    if alpha[i] > 0 and alpha[i] < self.C:
                        b = b1
                    elif alpha[j] > 0 and alpha[j] < self.C:
                        b = b2
                    else:
                        b = (b1 + b2) / 2.0
                    alphapairChanged += 1
                    print_count+=1
                    if(print_count% self.print_interval==0):
                        print("iter: %d i:%d,pairs changed %d" % (iter, i, alphapairChanged))
            if alphapairChanged == 0:
                iter += 1
            else:
                iter = 0
            if iter> iter_count:
                iter_count=iter
                print("iteration number: %d" % iter)

        self.alpha = alpha
        self.b = b
        self.data = dataMatrix
        self.label = labelMatrix
        return b, alpha

    def forward(self,input_feature):
        if np.ndim(input_feature) == 1:
            input_feature = np.expand_dims(input_feature,0)
        fxi = self.kernel.forward(self.alpha, self.label, self.data, self.b, input_feature)
        return fxi

    def selectJrand(self,i, m):
        """
        :param i: idx i
        :param m: the length of the data
        :return: a random number, where  j != i
        """
        j = i
        while j == i:
            j = int(random.uniform(0, m))
        return j

    def clipAlpha(self,a_j, H, L):
        """
        :param a_j: alpha_j
        :param H:  higher bound
        :param L: lower bound
        :return:  clipped alpha_j
        """
        if a_j > H:
            a_j = H
        if L > a_j:
            a_j = L
        return a_j

def test(data,label,model):
    """
    :param data: test data
    :param label: corresponding label
    :param model: trained model for testing
    :return: the predicted confidence list and accuracy
    """

    out_confidence = []
    out_pred = []
    for i in range(len(data)):
        predi = model.forward(data[i])
        out_confidence.append(predi)
        if predi>0:
            out_pred.append(1)
        else:
            out_pred.append(-1)

    count = 0
    for i in range(len(label)):
        if out_pred[i] == label[i]:
            count+=1

    acc = float(count)/len(label)
    out_confidence =np.array(out_confidence)
    out_confidence = out_confidence.squeeze(-1).squeeze(-1)

    out_pred = np.array(out_pred)
    return out_confidence,acc,out_pred


def norm_data(data):
    """
    linear norm for each data
    :param data:
    :return:
    """
    data = np.array(data)
    for i in range(data.shape[1]):
        data[:,i] =  (data[:,i] -min( data[:,i] ))/(max( data[:,i] )-min( data[:,i] ))
    return data

def norm_label(label):
    """
    change the label
    :param label: list of the label
    :return:  changed label
    """
    label = np.array(label)
    for i in range(len(label)):
        if label[i] <= 0:
            label[i] = -1
        else:
            label[i] = 1
    return label

def shuffle_Data(X,Y):
    """
    get the shuffled data
    :param X: training data
    :param Y: label
    :return: shuffled data and label
    """
    idxes = np.arange(len(X))
    random.shuffle(idxes)
    X = X[idxes]
    Y = Y[idxes]

    return X,Y

def SplitData(X,Y,train_rate=0.8):
    """
    :param X:  training data
    :param Y:  label
    :param train_rate: the rate for the training, 1-rate for testing
    :return: split data
    """
    len = int(X.shape[0]*train_rate)
    Train_X = X[0:len]
    Train_Y = Y[0:len]
    Test_X = X[len:]
    Test_Y = Y[len:]
    return Train_X,Train_Y,Test_X,Test_Y


def AUPRC(pred_,label):
    return average_precision_score(label,pred_ )

def f1(pred_,label):
    return f1_score(label, pred_, average='macro')


def show_hist(data,label):
    tmp_data = np.concatenate([data,np.expand_dims(label,-1)],-1)

    # plt.figure(figsize=(8, 8), dpi=60)
    attribute_num = tmp_data.shape[1]
    for i in range(attribute_num):
        axi = plt.subplot(1,attribute_num,i+1)
        axi.hist(tmp_data[:,i], )

    # plt.show()
    plt.tight_layout()
    plt.savefig("./hist.png")


def loadDataSet(filename,split_tag="\t"):
    fr = open(filename)
    data = []
    label = []
    for line in fr.readlines():
        lineAttr = line.strip().split(split_tag)
        data.append([float(x) for x in lineAttr[:-1]])
        label.append(float(lineAttr[-1]))
    return data,label

class Kernel_Manager():

    def get_kernel(self,kwargs={}):
        if kwargs["kernel"] == "linear":
            return Linear_Kernel()
        elif kwargs["kernel"] == "rbf":
            return RBF_Kernel(kwargs["sigma"])
        elif kwargs["kernel"] == "Polynomial":
            return Polynomial_Kernel(kwargs["degree"])
        else:
            print("Please use the correct configurations")
            exit(1)



