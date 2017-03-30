import sys
import argparse

import process_data
from linreg import Linear_Regression

import tensorflow as tf
seed=12
tf.set_random_seed(seed)

import numpy as np


def main(_):

	n_samples=1000
	train_size=500
	validation_size=0
	batch_size=500
	
	
	learning_rate=0.001 # learning rate
	opt_params=dict(learning_rate=learning_rate)
	param_str='/lr=%0.4f' %(learning_rate)

	training_epochs=100
	ckpt_freq=2000 # define check pointing frequency

	# import data
	states=process_data.read_data_sets(data_params,train_size=train_size,validation_size=validation_size)

	
	# define model
	model=Linear_Regression(Ns,batch_size,opt_params)
	
	
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
		for index in range(training_epochs): # run 100 epochs

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

			print(sess.run( model.loss, feed_dict={model.X: batch_X,model.Y: batch_Y}))
			print(sess.run( model.loss, feed_dict={model.X: states.test.data_X, model.Y: states.test.data_Y}) )
			print('---------')
		# Step 9: test model
		#print(sess.run(model.loss, feed_dict={model.X: states.test.data_X, model.Y: states.test.data_Y}) )

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
