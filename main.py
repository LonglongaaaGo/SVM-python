
import sys
from sklearn.model_selection import KFold
import pandas as pd
from sklearn.model_selection import train_test_split
from SVM import *


def basic_Test(data,label,kernel,C=0.6, toler=0.001, maxIter=40,fold_k=5):
    model = SVM(C=C, toler=toler, maxIter=maxIter, kernel=kernel)

    fold_k = fold_k
    kf_tool = KFold(n_splits=fold_k, shuffle=True, random_state=None)

    acc_list = []
    auprc_list = []
    f1_score_list = []
    for train_index, test_index in kf_tool.split(data):
        train_x, train_y = data[train_index], label[train_index]
        val_x, val_y = data[test_index], label[test_index]
        model.train(train_x, train_y)
        out_confidence, acc, out_pred = test(val_x, val_y, model)
        auprc_ = AUPRC(out_pred, val_y)
        f1_score = f1(out_pred, val_y)

        print("-----acc:", acc)
        print("-----auprc_:", auprc_)
        print("-----f1_score:", f1_score)
        # print("out_confidence", out_confidence)

        acc_list.append(acc)
        auprc_list.append(auprc_)
        f1_score_list.append(f1_score)

    mean_acc = np.average(acc_list)
    mean_auprc = np.average(auprc_list)
    mean_f1_score = np.average(f1_score_list)
    print(mean_acc)
    print(mean_auprc)
    print(mean_f1_score)
    return mean_acc,mean_auprc,mean_f1_score



def img_show(input, mean_acc_list,mean_auprc_list,mean_f1_list,title,x_label,y_label,file_name="test"):

    # set linewidth and plot
    # plt.plot(input, output, linewidth=2)
    plt.plot(input, mean_acc_list, color='green', label='Accuracy')
    plt.plot(input, mean_auprc_list, color='red', label='AUPRC')
    plt.plot(input, mean_f1_list, color='blue', label='F1-score')
    # set Title and label
    #Ablation study of swarm size for "+ file_name
    plt.title(title,fontsize=10)
    plt.xlabel(x_label, fontsize=10)
    plt.ylabel(y_label, fontsize=10)
    plt.legend()

    # tick size
    plt.tick_params(axis='both',labelsize=8)
    plt.show()
    # plt.savefig(file_name+".png")



def save_show(input, mean_acc_list,mean_auprc_list,mean_f1_list,title,x_label,y_label,file_name="test"):

    # set linewidth and plot
    # plt.plot(input, output, linewidth=2)
    plt.plot(input, mean_acc_list, color='green', label='Accuracy')
    plt.plot(input, mean_auprc_list, color='red', label='AUPRC')
    plt.plot(input, mean_f1_list, color='blue', label='F1-score')
    # set Title and label
    #Ablation study of swarm size for "+ file_name
    plt.title(title,fontsize=10)
    plt.xlabel(x_label, fontsize=10)
    plt.ylabel(y_label, fontsize=10)
    plt.legend()

    # tick size
    plt.tick_params(axis='both',labelsize=8)
    # plt.show()
    plt.savefig(file_name+".png")



def Abalation_study_C():
    data, label = loadDataSet('data_banknote_authentication.txt',split_tag=",")
    data = norm_data(data)
    label = norm_label(label)
    data,label = shuffle_Data(data,label)
    # show_hist(data,label)


    kernel_manger = Kernel_Manager()

    configs = [
        {"kernel":"linear","C":0.01,"toler":0.001,"maxIter":10,"fold_k":5},
        {"kernel": "linear", "C": 0.1, "toler": 0.001, "maxIter": 10, "fold_k": 5},
        {"kernel": "linear", "C": 0.3, "toler": 0.001, "maxIter": 10, "fold_k": 5},
        {"kernel": "linear", "C": 0.6, "toler": 0.001, "maxIter": 10, "fold_k": 5},
        {"kernel": "linear", "C": 0.9, "toler": 0.001, "maxIter": 10, "fold_k": 5},
        {"kernel": "linear", "C": 1.5, "toler": 0.001, "maxIter": 10, "fold_k": 5},
        {"kernel": "linear", "C": 3, "toler": 0.001, "maxIter": 10, "fold_k": 5},
    ]

    mean_acc_list =[]
    mean_auprc_list =[]
    mean_f1_list =[]
    iters = []
    for i in range(len(configs)):
        tp_cofig = configs[i]
        iters.append(tp_cofig["C"])
        print("now, the configuration is:",tp_cofig)
        kernel = kernel_manger.get_kernel(tp_cofig)
        mean_acc,mean_auprc,mean_f1_score = basic_Test(data,label,kernel,
                        C=tp_cofig["C"], toler=tp_cofig["toler"], maxIter=tp_cofig["maxIter"],fold_k=tp_cofig["fold_k"])
        mean_acc = mean_acc
        mean_acc_list.append(mean_acc)
        mean_auprc_list.append(mean_auprc)
        mean_f1_list.append(mean_f1_score)

    img_show(iters, mean_acc_list,mean_auprc_list,mean_f1_list,title="Ablation study of parameter C",
             x_label="The value of parameter C",y_label="Score",file_name="parameter C")




def Abalation_study_sigma():
    data, label = loadDataSet('data_banknote_authentication.txt',split_tag=",")
    data = norm_data(data)
    label = norm_label(label)
    data,label = shuffle_Data(data,label)
    # show_hist(data,label)

    kernel_manger = Kernel_Manager()

    configs = [
        {"kernel":"rbf", "sigma":0.0001,"C": 0.6,"toler":0.001,"maxIter":10,"fold_k":5},
        {"kernel": "rbf", "sigma":0.001, "C": 0.6, "toler": 0.001, "maxIter": 10, "fold_k": 5},
        {"kernel": "rbf", "sigma":0.01, "C": 0.6, "toler": 0.001, "maxIter": 10, "fold_k": 5},
        {"kernel": "rbf", "sigma":0.1, "C": 0.6, "toler": 0.001, "maxIter": 10, "fold_k": 5},
        {"kernel": "rbf",  "sigma":1,"C": 0.6, "toler": 0.001, "maxIter": 10, "fold_k": 5},
        {"kernel": "rbf", "sigma":5, "C": 0.6, "toler": 0.001, "maxIter": 10, "fold_k": 5},
        {"kernel": "rbf",  "sigma":10,"C": 0.6, "toler": 0.001, "maxIter": 10, "fold_k": 5},
    ]

    mean_acc_list =[]
    mean_auprc_list =[]
    mean_f1_list =[]
    iters = []
    for i in range(len(configs)):
        tp_cofig = configs[i]
        iters.append(tp_cofig["sigma"])
        print("now, the configuration is:",tp_cofig)
        kernel = kernel_manger.get_kernel(tp_cofig)
        mean_acc,mean_auprc,mean_f1_score = basic_Test(data,label,kernel,
                        C=tp_cofig["C"], toler=tp_cofig["toler"], maxIter=tp_cofig["maxIter"],fold_k=tp_cofig["fold_k"])
        mean_acc = mean_acc
        mean_acc_list.append(mean_acc)
        mean_auprc_list.append(mean_auprc)
        mean_f1_list.append(mean_f1_score)

    img_show(iters, mean_acc_list,mean_auprc_list,mean_f1_list,title="Ablation study of the parameter sigma",
             x_label="The value of the parameter sigma",y_label="Score",file_name="parameter sigma")



def Abalation_study_degree():
    data, label = loadDataSet('data_banknote_authentication.txt',split_tag=",")
    data = norm_data(data)
    label = norm_label(label)
    data,label = shuffle_Data(data,label)
    # show_hist(data,label)

    kernel_manger = Kernel_Manager()

    configs = [
        {"kernel": "Polynomial", "degree": 0.001, "C": 0.6, "toler": 0.001, "maxIter": 10, "fold_k": 5},
        {"kernel": "Polynomial", "degree":0.01, "C": 0.6, "toler": 0.001, "maxIter": 10, "fold_k": 5},
        {"kernel": "Polynomial", "degree":0.1, "C": 0.6, "toler": 0.001, "maxIter": 10, "fold_k": 5},
        {"kernel": "Polynomial",  "degree":1,"C": 0.6, "toler": 0.001, "maxIter": 10, "fold_k": 5},
        {"kernel": "Polynomial", "degree":2, "C": 0.6, "toler": 0.001, "maxIter": 10, "fold_k": 5},
        {"kernel": "Polynomial",  "degree":5,"C": 0.6, "toler": 0.001, "maxIter": 10, "fold_k": 5},
        {"kernel": "Polynomial", "degree": 10, "C": 0.6, "toler": 0.001, "maxIter": 10, "fold_k": 5},
    ]

    mean_acc_list =[]
    mean_auprc_list =[]
    mean_f1_list =[]
    iters = []
    for i in range(len(configs)):
        tp_cofig = configs[i]
        iters.append(tp_cofig["degree"])
        print("now, the configuration is:",tp_cofig)
        kernel = kernel_manger.get_kernel(tp_cofig)
        mean_acc,mean_auprc,mean_f1_score = basic_Test(data,label,kernel,
                        C=tp_cofig["C"], toler=tp_cofig["toler"], maxIter=tp_cofig["maxIter"],fold_k=tp_cofig["fold_k"])
        mean_acc = mean_acc
        mean_acc_list.append(mean_acc)
        mean_auprc_list.append(mean_auprc)
        mean_f1_list.append(mean_f1_score)

    img_show(iters, mean_acc_list,mean_auprc_list,mean_f1_list,title="Ablation study of the degree",
             x_label="The value of the parameter degree",y_label="Score",file_name="parameter degree")


if __name__ == '__main__':
    Abalation_study_degree()