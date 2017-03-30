# -*- coding: utf-8 -*-
import sys, os

import tensorflow as tf
#seed=12
#tf.set_random_seed(seed)

import numpy as np


class Linear_Regression:
	# build the graph for the model
	def __init__(self,n_feats,n_samples,opt_params):

		# define global step for checkpointing
		self.global_step=tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
		self.n_feats=n_feats
		self.n_samples=n_samples

		# Step 1: create placeholders for input X and label Y
		self._create_placeholders()
		# Step 2: create weight and bias, initialized to 0 and construct model to predict Y from X
		self._create_model()
		# Step 3: define loss function
		self._create_loss()
		self._create_accuracy()
		# Step 4: use gradient descent to minimize loss
		self._create_optimiser(opt_params)
		# Step 5: create sumamries
		self._create_summaries()


	def _create_placeholders(self):
		with tf.name_scope('data'):
			self.X=tf.placeholder(tf.float32, shape=(None,self.n_feats), name="X_data")
			self.Y=tf.placeholder(tf.float32, shape=(None,1), name="Y_data")

	def _create_model(self):
		with tf.name_scope('model'):
			self.J=tf.Variable( tf.zeros((self.n_feats,self.n_feats), ),dtype=tf.float32, name="int")
			self.h=tf.Variable( tf.zeros((self.n_feats,), ),dtype=tf.float32, name="field")
			# define model
			self.Y_predicted=0.5*tf.einsum('ai,ij,aj->a',self.X,self.J,self.X) \
#							   + tf.einsum('ai,i->a',self.X,self.h)
			#print(self.Y_predicted.shape)
			
	def _create_loss(self):
		with tf.name_scope('loss'):
			#self.loss = tf.square(self.Y - self.Y_predicted, name="loss")
			self.loss = tf.reduce_sum(tf.pow(self.Y - self.Y_predicted, 2))/(2.0*self.n_samples) \
						+ 0.1*tf.reduce_sum(tf.abs(self.J))
			#self.loss = tf.reduce_mean( tf.nn.l2_loss(self.Y - self.Y_predicted))

	def _create_accuracy(self):
		with tf.name_scope('accuracy'):
			pass

	def _create_optimiser(self,kwargs):
		with tf.name_scope('optimiser'):
			#self.optimizer = tf.train.GradientDescentOptimizer(**kwargs).minimize(self.loss,global_step=self.global_step)
			self.optimizer = tf.train.AdamOptimizer(**kwargs).minimize(self.loss,global_step=self.global_step)


	def _create_summaries(self):
		with tf.name_scope("summaries"):
			tf.summary.scalar("loss", self.loss)
			#tf.summary.scalar("accuracy", self.accuracy)
			tf.summary.histogram("histogram loss", self.loss)
			# merge all summaries into one op to make it easier to manage
			self.summary_op = tf.summary.merge_all()

