# SVM-python
 The implementation of Support vector machine (SVM) using python-numpy.

Any nation popularly uses banknotes to carry-out financial activities. However, a lot of fake notes are produced in the market without legal sanction, and hinder the development of the enconomy of the world. Thus, it is very important for us to presicely detect the forgery banknotes. In this project, we perform the banknote authentication on public banknote authentication dataset by using support vector machine algorithm. The banknote dataset involves predicting whether a given banknote is authentic given a number of measures taken from a photograph. It contains 1,372 rows with 5 numeric variables. Exhaustive experiments have been conducted using different hyper-parameters for find the best configurations for support vector machine algorithm. We find that the support vector machine can perform well with 100\% accuracy on this task.

# Implementation
We use the Python to implement the support vector machine algorithm. For solving the final objective function, we use the SMO algorithm to find the optimized weights. Unless specified, we train the support vector machine with iterations of 40, $C$ parameter of 0.6, tolerance of 0.001, and conduct a 5-fold cross-validation for each experiment.


# Banknote authentication dataset
The dataset used for carrying out the experiments is taken from UCI machine learning repository [https://archive.ics.uci.edu/ml/machine-learning data bases/00267/data_banknote_authentication.txt](data_banknote_authentication.txt). The dataset is owned by Volker Lohweg (University of Applied Sciences). The dataset has a total of 1372 instances. Each instance has five attributes, out of which four attributes are real-valued attributes, and one attribute is a corresponding label. Genuine banknotes labeled as 0, account for 55\% of the whole data, and the fake banknotes, represented as 1, account for 45\%.
