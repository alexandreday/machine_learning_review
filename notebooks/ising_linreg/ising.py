import sys
import argparse

import process_data
from linreg import Linear_Regression

import tensorflow as tf
seed=12 #None
tf.set_random_seed(seed)

import numpy as np
import matplotlib.pyplot as plt
#np.set_printoptions(threshold=np.nan)


def main(_):

	symmetric=True # interaction coupling symemtry
	J_const=False # constant nn interaction
	nn=False # nearest-neighbour only


	training_epochs=1000
	ckpt_freq=2000000 # define inverse check pointing frequency

	n_samples=40000
	train_size=300
	noise_width=0.0
	validation_size=0
	batch_size=10
	
	# GD/ADAM learning params
	learning_rate=0.0001 #0.000002 # learning rate
	beta1=0.9 #0.94
	beta2=0.99 #0.87
	epsilon=1e-09

	#opt_params=dict(learning_rate=learning_rate,beta1=beta1,beta2=beta2,epsilon=epsilon)
	opt_params=dict(learning_rate=learning_rate)
	param_str='/lr=%0.4f' %(learning_rate)

	# import data
	data_params['nn']=nn
	states=process_data.read_data_sets(data_params,train_size=train_size,
										validation_size=validation_size,noise_width=noise_width)

	# define model
	model=Linear_Regression(Lx,opt_params,J_const=J_const,nn=nn,symmetric=symmetric)


	"""
	X = states.train.data_X
	Y = states.train.data_Y
	Jlinreg=np.dot( np.linalg.pinv(X.T.dot(X) ), np.dot(X.T, Y) )

	'''
	print(X.shape)
	_, s, _ = np.linalg.svd(X, full_matrices=True)
	print(s.shape)

	plt.hist(s,bins=100)
	plt.show()

	exit()
	'''

	print(Jlinreg)

	Y_test_predicted = np.dot( states.test.data_X,Jlinreg)
	Y_test = states.test.data_Y
	
	plt.scatter(Y_test,Y_test_predicted)
	plt.show()

	
	plt.hist(Jlinreg,bins=50)
	plt.show()

	exit()
	"""
	
	saver = tf.train.Saver() # defaults to saving all variables
	with tf.Session() as sess:

		"""
		# restore most recent session from checkpoint directory
		ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/checkpoint'))
		if ckpt and ckpt.model_checkpoint_path:
	 		saver.restore(sess, ckpt.model_checkpoint_path) 
		"""

		# Step 7: initialize the necessary variables, in this case, w and b
		sess.run(tf.global_variables_initializer())

		average_loss = 0.0
		# write summary
		#writer = tf.summary.FileWriter('./ising_reg'+param_str, sess.graph)

		# Step 8: train the model
		for index in range(training_epochs): 

			batch_X, batch_Y = states.train.next_batch(batch_size,seed=seed)
			
			loss_batch, _, summary = sess.run([model.loss, model.optimizer, model.summary_op],
												feed_dict={model.X: batch_X,model.Y: batch_Y} )
			# count training step
			step = sess.run(model.global_step)
			
			# add summary data to writer
			#writer.add_summary(summary, global_step=step)

			average_loss += loss_batch
			if (index + 1) % ckpt_freq == 0:
				saver.save(sess, './checkpoints/ising_reg', global_step=step)

			J = sess.run(model.J)
			print("loss:", index, sess.run( model.loss/batch_size, feed_dict={model.X: batch_X,model.Y: batch_Y}) )
			print("Jmin,Jmax,Jmean,Jjj:", np.max(J), np.min(J), np.mean(J), np.sum(J[[i*101 for i in range(100)]]) )


		# Step 9: test model
		J = sess.run(model.J)
		print(sess.run( model.loss/(n_samples-train_size), feed_dict={model.X: states.test.data_X, model.Y: states.test.data_Y}) )
		print("J:", np.max(J), np.min(J), np.mean(J) )
		

		plt.hist(J,bins=100)
		plt.show()

		print("\sum_j J_{jj}=", np.sum(J[[i*101 for i in range(100)]]) )

		plt.hist(J[[i*101 for i in range(100)]],bins=100)
		plt.show()

		Y,Y_predicted=sess.run([model.Y,model.Y_predicted], feed_dict={model.X: states.test.data_X, model.Y: states.test.data_Y})		

		plt.scatter(Y,Y_predicted)
		plt.show()

		Jij=J.reshape((Lx,Ly,Lx,Ly))

		cmap_args=dict(vmin=-2., vmax=2., cmap='seismic')		

		plt.imshow(Jij.reshape(Lx*Ly,Lx*Ly),**cmap_args)
		plt.colorbar()
		plt.show()

		plt.imshow(Jij[:,4,:,4],**cmap_args)
		plt.colorbar()
		plt.show()

		plt.pcolor(Jij[4,:,4,:],**cmap_args)
		plt.colorbar()
		plt.show()

		exit()


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
	tf.app.run(main=main, argv=[sys.argv[0]] )
