from sklearn.datasets import fetch_mldata

import numpy as np

import seaborn as sns

#download mnist dataset
mnist = fetch_mldata('MNIST original')

#generate a smaller subset
train_subset = mnist.data[5::100]

#generate a smaller cross validation subset
cval_subset = mnist.data[50::1000]

#generate a subset of y values
#NOTE: the subset has to be of the same shape as the of the training data!!!
t_subset = mnist.target[5::100]

sns.distplot(t_subset, bins=[0,1,2,3,4,5,6,7,8,9,10])

#Same for the cross validation y 
cval_target_subset = mnist.target[50::1000]

#We have to transform the numbers 0 to 9 into a one versus all classification y vector
#create an empty matrix with the length of the trainings subset
new_target = np.zeros((700, 10))

#we loop over the trainings y, and set the nth value in the one versus all y vector to 1
#e.g. if our target is 5, we set the 5th entry in the y vector to 1
#this gives us a matrix of ones and 0 as a target
it = np.nditer(t_subset, flags=['f_index'])

while not it.finished:
    #print('item, index ', it[0], it.index)
    new_target[it.index, int(it[0])] = 1
    it.iternext()


#Set up the X and y values used in the main loop (for simplicity of notation)
X = train_subset
y = new_target
X.shape
y.shape

# sigmoid function
def sigmoid(x):
    return 1/(1+np.exp(-x))

#sigmoid derivative function  
def sigmoid_derivative(x):
    return sigmoid(x)*(1-sigmoid(x))
    


# randomly initialize our weights with mean 0
syn0 = 2*np.random.random((784,400)) - 1
syn1 = 2*np.random.random((400,400)) - 1
syn2 = 2*np.random.random((400,10)) - 1

alpha = 0.05
for j in range(6000):

	# Feed forward through layers 0, 1, and 2
    l0 = X
    l1 = sigmoid(np.dot(l0,syn0))
    l2 = sigmoid(np.dot(l1,syn1))
    l3 = sigmoid(np.dot(l2,syn2))

    # how much did we miss the target value?
    l3_error = y - l3
    
    if (j% 100) == 0:
        print ("Error:" + str(np.mean(np.abs(l3_error))))
        
    # in what direction is the target value?
    # were we really sure? if so, don't change too much.
    l3_delta = l3_error*sigmoid_derivative(l3)

    # how much did each l1 value contribute to the l2 error (according to the weights)?
    l2_error = l3_delta.dot(syn2.T)
    
    # in what direction is the target l1?
    # were we really sure? if so, don't change too much.
    l2_delta = l2_error * sigmoid_derivative(l2)
    
    # how much did each l1 value contribute to the l2 error (according to the weights)?
    l1_error = l2_delta.dot(syn1.T)
    
    # in what direction is the target l1?
    # were we really sure? if so, don't change too much.
    l1_delta = l1_error * sigmoid_derivative(l1)

    syn2 += l2.T.dot(l3_delta)*alpha
    syn1 += l1.T.dot(l2_delta)*alpha
    syn0 += l0.T.dot(l1_delta)*alpha
    


plot_x = [0,1,2,3,4,5,6,7,8,9]

sns.barplot(x=plot_x, y=l3[455])

sample = X[601]
smat = np.split(sample,28)
sns.heatmap(smat,square=True)

######################################################
# Cval
#######################################################

#cross validation
it = np.nditer(cval_target_subset, flags=['f_index'])

cval_target = np.zeros((70, 10))
while not it.finished:
    #print('item, index ', it[0], it.index)
    cval_target[it.index, int(it[0])] = 1
    it.iternext()

#cross validation feed forward
y_c = cval_target
l0_c = cval_subset
l1_c = sigmoid(np.dot(l0_c,syn0))
l2_c = sigmoid(np.dot(l1_c,syn1))
l3_c = sigmoid(np.dot(l2_c,syn2))
l3_error_c =  y_c - l3_c
print ("Accuracy:", 1- np.mean(np.abs(l3_error_c)))

