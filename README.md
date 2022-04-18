# SVM-python
 The implementation of Support vector machine (SVM) using python-numpy.

Any nation popularly uses banknotes to carry-out financial activities. However, a lot of fake notes are produced in the market without legal sanction, and hinder the development of the enconomy of the world. Thus, it is very important for us to presicely detect the forgery banknotes. In this project, we perform the banknote authentication on public banknote authentication dataset by using support vector machine algorithm. The banknote dataset involves predicting whether a given banknote is authentic given a number of measures taken from a photograph. It contains 1,372 rows with 5 numeric variables. Exhaustive experiments have been conducted using different hyper-parameters for find the best configurations for support vector machine algorithm. We find that the support vector machine can perform well with 100\% accuracy on this task.

# Banknote authentication dataset
The dataset used for carrying out the experiments is taken from UCI machine learning repository [data_banknote_authentication.txt]. The dataset is owned by Volker Lohweg (University of Applied Sciences). The dataset has a total of 1372 instances. Each instance has five attributes, out of which four attributes are real-valued attributes, and one attribute is a corresponding label. Genuine banknotes labeled as 0, account for 55\% of the whole data, and the fake banknotes, represented as 1, account for 45\%.
- data_banknote_authentication.txt: https://archive.ics.uci.edu/ml/machine-learning data bases/00267/data_banknote_authentication.txt

# Implementation
We use the Python to implement the support vector machine algorithm. For solving the final objective function, we use the SMO algorithm to find the optimized weights. Unless specified, we train the support vector machine with iterations of 40, C parameter of 0.6, tolerance of 0.001, and conduct a 5-fold cross-validation for each experiment.

## Pre-prosessing
![Image](./hist.png#pic_center)

As shown in Fig.[hist](./hist.png), since each attribute has different means and standard deviations,  we apply the linear normalization for better modeling. Moreover, we change labels with 0 to the value -1 for the convenience of the SVM training. 


![Image](./parameter_C.png#pic_center)


## Abalation study of the parameter C
In this subsection, we use the linear kernel to explore the influence of parameter C in a range from 0.001 to 3.0. As shown in Fig.[parameter_C](./parameter_C.png), with the increase of the parameter C, the model performance has a upward trend. However, when the value of C is higher than 0.5, the performance does not change too much.


![Image](./parameter_degree.png#pic_center)


## Abalation study of the parameter degree
Although the linear kernel can achieve around 97\% accuracy in this task, there exists much room for improvement. Since the linear kernel cannot learn the non-linear correlation of data, we further apply the polynomial kernel to train the model. As shown in Fig.[parameter_degree](./parameter_degree.png), we perform the ablation for the degree in the polynomial kernel. We conduct the 5-fold cross-validation for the degree with the range from 0.0001 to 10. We can find that we can get the performance gains when increasing the degrees of the polynomial kernel. If the degree is near zero, the performance of recognition decrease rapidly. 


![Image](./parameter_sigma.png#pic_center)


# Abalation study of the parameter sigma
We use the Gaussian kernel for leanring non-linear representations in the dataset for better performance. Thus, we conduct the ablation study of the parameter sigma in the range from 0.0001 to 10. The ablation is shown in Fig.[parameter_sigma](./parameter_sigma.png), we can find that when using small value of the sigma, the performance is not good and even the accuracy of this configuration is lower than model using linear kernel. Fortunately, when the sigma is higher than 5.0, the model can achive the best performance with  accuracy of 100\%,  F1-score of 1.0, and AUPRC of 1.0, respectively.
