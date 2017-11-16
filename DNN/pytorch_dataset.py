from __future__ import print_function
import torch # pytorch package, allows using GPUs
from torchvision import datasets, transforms # load MNIST data
import os,sys

import torch # pytorch package, allows using GPUs
from torch.autograd import Variable # differentiation of pytorch tensors

import numpy as np




class IsingDataset(torch.utils.data.Dataset):
	"""2D Ising model pytorch dataset."""

	def __init__(self, pkl_file, root_dir, transform=None):
		"""
		Args:
			csv_file (string): Path to the csv file with annotations.
			root_dir (string): Directory with all the images.
			transform (callable, optional): Optional transform to be applied
				on a sample.
		"""

		import pickle

		L=40 # linear system size
		T=np.linspace(0.25,4.0,16) # temperatures
		T_c=2.26 # critical temperature in the TD limit

		# preallocate ordered, critical and disordered states
		X_ordered=np.zeros((90000,L**2),dtype=np.int32)
		Y_ordered=np.ones((90000,),dtype=np.int32)

		X_disordered=np.zeros((80000,L**2),dtype=np.int32)
		Y_disordered=np.zeros((80000,),dtype=np.int32)

		# load data in ordered phase
		for i,T_i in enumerate(T[:9]):

			file_name='mag_vs_T_L%i_T=%.2f.pkl' %(L,T_i)
			
			f=open(root_dir+pkl_file,'rb')
			X_ordered[10000*i:10000*(i+1),:]=np.unpackbits(pickle.load(f)).reshape(-1, L**2)


		# load data in disordered phase
		for i,T_i in enumerate(T[9:]):

			file_name='mag_vs_T_L%i_T=%.2f.pkl' %(L,T_i)
			
			f=open(root_dir+pkl_file,'rb')
			X_disordered[10000*i:10000*(i+1),:]=np.unpackbits(pickle.load(f)).reshape(-1, L**2)

		# map 0 state to -1 (Ising variable can take values $\pm 1$)
		X_ordered[np.where(X_ordered==0)]=-1
		X_disordered[np.where(X_disordered==0)]=-1

		X_data=np.concatenate((X_ordered,X_disordered))
		Y_data=np.concatenate((Y_ordered,Y_disordered))

		del X_ordered, X_disordered

		
		self.Ising_data = (X_data.reshape(-1,L,L), Y_data)
		self.root_dir = root_dir
		self.transform = transform

	
	def __len__(self):
		return len(self.Ising_data[1])

	def __getitem__(self, idx):

		sample=(self.Ising_data[0][idx,:,:],self.Ising_data[1][idx])

		if self.transform:
			sample=self.transform(sample)

		return sample




pkl_file='mag_vs_T_L40_T=3.50.pkl'
root_dir=os.path.expanduser('~')+'/Dropbox/MachineLearningReview/Datasets/isingMC/compressed/'

dataset=IsingDataset(pkl_file,root_dir)

print(len(dataset))


train_loader = torch.utils.data.DataLoader(
    dataset,batch_size=20, shuffle=True)


# oop over training data
for batch_idx, (data, label) in enumerate(train_loader):
	# wrap minibatch data in Variable
	data, label = Variable(data), Variable(label)
	print(batch_idx)


