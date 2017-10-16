from __future__ import print_function
print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
#import seaborn

import process_data

from sklearn import linear_model


def main():

	nn=False # nearest-neighbour only


	training_epochs=1000
	ckpt_freq=2000000 # define inverse check pointing frequency

	n_samples=1000
	train_size=30000
	noise_width=0.0

	# import data
	data_params['nn']=nn
	states=process_data.read_data_sets(data_params,train_size=train_size,noise_width=noise_width)


	X_train=states.train.data_X[:n_samples]
	Y_train=states.train.data_Y[:n_samples].squeeze()
	X_test=states.test.data_X[:n_samples]
	Y_test=states.test.data_Y[:n_samples].squeeze()

	#Set up Lasso and Ridge Regression models
	leastsq=linear_model.LinearRegression()
	ridge=linear_model.Ridge()
	lasso = linear_model.Lasso()

	alphas = [1E4]# np.logspace(-2, 2, 8)

	train_errors_leastsq = list()
	test_errors_leastsq = list()

	train_errors_ridge = list()
	test_errors_ridge = list()

	train_errors_lasso = list()
	test_errors_lasso = list()



	#Initialize coeffficients for ridge regression and Lasso
	coefs_leastsq = []
	coefs_ridge = []
	coefs_lasso=[]

	for a in alphas:

		### ordinary least squares
		leastsq.fit(X_train, Y_train) # fit model 
		coefs_leastsq.append(leastsq.coef_) # store weights
		# use the coefficient of determination R^2 as the performance of prediction.
		train_errors_leastsq.append(leastsq.score(X_train, Y_train))
		test_errors_leastsq.append(leastsq.score(X_test,Y_test))

		### apply Ridge regression
		ridge.set_params(alpha=a) # set regularisation parameter
		ridge.fit(X_train, Y_train) # fit model 
		coefs_ridge.append(ridge.coef_) # store weights
		# use the coefficient of determination R^2 as the performance of prediction.
		train_errors_ridge.append(ridge.score(X_train, Y_train))
		test_errors_ridge.append(ridge.score(X_test,Y_test))

		### apply Ridge regression
		lasso.set_params(alpha=a) # set regularisation parameter
		lasso.fit(X_train, Y_train) # fit model
		coefs_lasso.append(lasso.coef_) # store weights
		# use the coefficient of determination R^2 as the performance of prediction.
		train_errors_lasso.append(lasso.score(X_train, Y_train))
		test_errors_lasso.append(lasso.score(X_test,Y_test))


		#"""
		J_leastsq=np.array(leastsq.coef_).reshape((Lx,Ly,Lx,Ly))
		J_ridge=np.array(ridge.coef_).reshape((Lx,Ly,Lx,Ly))
		J_lasso=np.array(lasso.coef_).reshape((Lx,Ly,Lx,Ly))

		cmap_args=dict(vmin=-1., vmax=1., cmap='seismic')

		fig, axarr = plt.subplots(nrows=1, ncols=3)
		axarr[0].imshow(J_leastsq.reshape(Lx*Ly,Lx*Ly),**cmap_args)
		axarr[0].set_title('$\\mathrm{OLS},\ \\lambda=%.2f$' %(a),fontsize=18)
		axarr[1].imshow(J_ridge.reshape(Lx*Ly,Lx*Ly),**cmap_args)
		axarr[1].set_title('$\\mathrm{Ridge},\ \\lambda=%.2f$' %(a),fontsize=18)
		im=axarr[2].imshow(J_lasso.reshape(Lx*Ly,Lx*Ly),**cmap_args)
		axarr[2].set_title('$\\mathrm{Lasso},\ \\lambda=%.2f$' %(a),fontsize=18)

		divider = make_axes_locatable(axarr[2])
		cax = divider.append_axes("right", size="5%", pad=0.05)
		fig.colorbar(im, cax=cax)

		#fig.subplots_adjust(right=2.0)

		plt.show()
    
		#"""



	###############################################################################
	# Display results

	# First see how the 10 features we learned scale as we change the regularization parameter
	plt.subplot(1,2,1)
	plt.semilogx(alphas, np.abs(coefs_ridge))
	axes = plt.gca()
	#ax.set_xscale('log')
	#ax.set_xlim(ax.get_xlim()[::-1])  # reverse axis
	plt.xlabel(r'$\lambda$',fontsize=18)
	plt.ylabel('$|w_i|$',fontsize=18)
	plt.title('Ridge')
	

	plt.subplot(1,2,2)
	plt.semilogx(alphas, np.abs(coefs_lasso))
	axes = plt.gca()
	#ax.set_xscale('log')
	#ax.set_xlim(ax.get_xlim()[::-1])  # reverse axis
	plt.xlabel(r'$\lambda$',fontsize=18)
	#plt.ylabel('$|\mathbf{w}|$',fontsize=18)
	plt.title('LASSO')
	plt.show()



	# Plot our performance on both the training and test data
	plt.semilogx(alphas, train_errors_ridge, 'b',label='Train (Ridge)')
	plt.semilogx(alphas, test_errors_ridge, 'g',label='Test (Ridge)')
	plt.semilogx(alphas, train_errors_lasso, '--b',label='Train (LASSO)')
	plt.semilogx(alphas, test_errors_lasso, '--g',label='Test (LASSO)')
	plt.semilogx(alphas, train_errors_lasso, 'xb',label='Train (OLS)')
	plt.semilogx(alphas, test_errors_lasso, 'xg',label='Test (OLS)')

	#plt.vlines(alpha_optim, plt.ylim()[0], np.max(test_errors), color='k',
	#           linewidth=3, label='Optimum on test')
	plt.legend(loc='upper right')
	plt.ylim([0, 1.0])
	plt.xlabel(r'$\lambda$',fontsize=18)
	plt.ylabel('Performance')
	plt.show()


if __name__ == '__main__':

	### define Ising model aprams
	# system size
	Lx=10
	Ly=10
	Ns=Lx*Ly
	# temperature
	T=3.0 

	# data dict
	data_params=dict(L=Lx,T=T)

	# run ML tool
	main()