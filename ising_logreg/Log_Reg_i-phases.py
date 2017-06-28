import numpy as np
import matplotlib.pyplot as plt
#import seaborn
import pickle
import os

np.random.seed()

from sklearn import utils, linear_model
from sklearn.neural_network import MLPClassifier

np.set_printoptions(threshold=np.nan)

# ML parameters
n_train=60000
n_test=10000

# physics model params
L=40 # system size
J=-1.0 # Ising interaction
T=np.linspace(0.25,4.0,16) # temperatures

# load data

path_to_data=os.path.expanduser('~')+'/Dropbox/MachineLearningReview/Datasets/isingMC/pickled/'


X_ordered=np.zeros((70000,L**2),dtype=np.int32)
Y_ordered=np.ones((70000,),dtype=np.int32)

X_critical=np.zeros((30000,L**2),dtype=np.int32)
Y_critical=np.ones((30000,),dtype=np.int32)

X_disordered=np.zeros((60000,L**2),dtype=np.int32)
Y_disordered=np.zeros((60000,),dtype=np.int32)

# load data in ordered phase
for i,T_i in enumerate(T[:7]):

	file_name='mag_vs_T_L%i_T=%.2f.pkl' %(L,T_i)
	
	f=open(path_to_data+file_name,'rb')
	X_ordered[10000*i:10000*(i+1),:]=pickle.load(f)

# load data in critical region
for i,T_i in enumerate(T[7:10]):

	file_name='mag_vs_T_L%i_T=%.2f.pkl' %(L,T_i)

	f=open(path_to_data+file_name,'rb')
	X_critical[10000*i:10000*(i+1),:]=pickle.load(f)
	if T_i>2.4:
		Y_critical[10000*i:10000*(i+1)]=0


# load data in disordered phase
for i,T_i in enumerate(T[10:]):

	file_name='mag_vs_T_L%i_T=%.2f.pkl' %(L,T_i)
	
	f=open(path_to_data+file_name,'rb')
	X_disordered[10000*i:10000*(i+1),:]=pickle.load(f)

X_ordered[np.where(X_ordered==0)]=-1
X_disordered[np.where(X_disordered==0)]=-1
X_critical[np.where(X_critical==0)]=-1

# pick random data points from ordered and disordered data
inds_ordered=np.random.random_integers(0,X_ordered.shape[0]-1,n_train+n_test)
inds_disordered=np.random.random_integers(0,X_disordered.shape[0]-1,n_train+n_test)

# define train and test data
X_train = np.concatenate((X_ordered[inds_ordered[:n_train]],X_disordered[inds_disordered[:n_train]]))
Y_train = np.concatenate((Y_ordered[inds_ordered[:n_train]],Y_disordered[inds_disordered[:n_train]]))
# shuffle data
X_train, Y_train = utils.shuffle(X_train, Y_train, random_state=0)


X_test  = np.concatenate((X_ordered[inds_ordered[n_train:]],X_disordered[inds_disordered[n_train:]]))
Y_test  = np.concatenate((Y_ordered[inds_ordered[n_train:]],Y_disordered[inds_disordered[n_train:]]))

#print(X_train.shape)
#print(Y_train)


'''
for i in range(X_train.shape[0]):
	plt.matshow(X_train[i,:].reshape(L,L))
	plt.title('phase %i' %(Y_train[i]) )
	plt.show()
exit()
'''

print('finished processing data')

# define logistic regressor
logreg=linear_model.LogisticRegression(C=1E3,random_state=1,verbose=0,max_iter=1E3,tol=1E-5) #penalty='l1'

# fit data
logreg.fit(X_train, Y_train)

# check accuracy
train_accuracy=logreg.score(X_train,Y_train)
test_accuracy=logreg.score(X_test,Y_test)
critical_accuracy=logreg.score(X_critical,Y_critical)

print(train_accuracy,test_accuracy,critical_accuracy)


clf = linear_model.SGDClassifier(loss='log', penalty='l2', alpha=5E0, n_iter=100, shuffle=True, random_state=1, learning_rate='optimal')
clf.fit(X_train,Y_train)
train_accuracy=clf.score(X_train,Y_train)
test_accuracy=clf.score(X_test,Y_test)
critical_accuracy=clf.score(X_critical,Y_critical)

print(train_accuracy,test_accuracy,critical_accuracy)



