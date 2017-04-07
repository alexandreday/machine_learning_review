import sys
import argparse

import process_data
from linreg import Linear_Regression

import tensorflow as tf
seed=12
tf.set_random_seed(seed)

import numpy as np
#np.set_printoptions(threshold=np.nan)


def main(_):

	symmetric=True # interaction coupling symemtry
	J_const=False # constant nn interaction
	nn=False # nearest-neighbour only


	training_epochs=2000
	ckpt_freq=200000 # define inverse check pointing frequency

	n_samples=1000
	train_size=800
	validation_size=0
	batch_size=100
	
	# ADAM learning params
	learning_rate=0.001 # learning rate
	beta1=0.9
	beta2=0.9999
	epsilon=1e-09

	opt_params=dict(learning_rate=learning_rate,beta1=beta1,beta2=beta2,epsilon=epsilon)
	#opt_params=dict(learning_rate=learning_rate)
	param_str='/lr=%0.4f' %(learning_rate)

	# import data
	data_params['nn']=nn
	states=process_data.read_data_sets(data_params,train_size=train_size,validation_size=validation_size)

	# define model
	model=Linear_Regression(Lx,opt_params,J_const=J_const,nn=nn,symmetric=symmetric)
	
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
			#print(batch_X.shape)
			#exit()

			loss_batch, _, summary = sess.run([model.loss, model.optimizer, model.summary_op],
												feed_dict={model.X: batch_X,model.Y: batch_Y} )
			# count training step
			step = sess.run(model.global_step)
			
			# add summary data to writer
			#writer.add_summary(summary, global_step=step)

			average_loss += loss_batch
			if (index + 1) % ckpt_freq == 0:
				saver.save(sess, './checkpoints/ising_reg', global_step=step)

			print("loss:", sess.run( model.loss, feed_dict={model.X: batch_X,model.Y: batch_Y}) )
	

		# Step 9: test model
		print(sess.run( model.loss, feed_dict={model.X: states.test.data_X, model.Y: states.test.data_Y}) )
		print("J:", sess.run(model.J) )			



if __name__ == '__main__':

	### define Ising model aprams
	# system size
	Lx=20
	Ly=20
	Ns=Lx*Ly
	# temperature
	T=4.0 

	# data dict
	data_params=dict(L=Lx,T=T)

	# run ML tool
	tf.app.run(main=main, argv=[sys.argv[0]] )
