import numpy as np
import torch
import torch.nn as nn
from scipy.special import softmax
import sys
import os
import torch.utils.data as Data
from torch.autograd import Variable

class SLCD(nn.Module):
	def __init__(self):
		super(SLCD,self).__init__()
		self.fc1 = nn.Linear(in_features=6, out_features=6, bias=True)
		self.relu = nn.ReLU()
		self.fc2 = nn.Linear(in_features=6, out_features=6, bias=True)
		self.softmax = nn.Softmax(dim=1)

	def forward(self, x):
		x1 = self.fc1(x)
		x2 = self.relu(x1)
		x3 = self.fc2(x2)
		x4 = self.softmax(x3)
		return x4

def default_seq_reader(videoslist, length, stride):
		sequences = []
		labels = []
		num_sequences = 0
		for videos in videoslist:
				video_length = len(videos)
				if (video_length < length):
						continue
				
				for i in range(0, video_length - length, stride):
						seq = videos[i: i + length]
						if (len(seq) == length):
								sequences.append(seq)
								lab = []
								for sq in seq:
										lab.append(float(sq.split(" ")[1]))
						labels.append(max(lab))
						num_sequences = num_sequences + 1
		return sequences, labels


def default_list_reader(fileList):
		videos = []
		for sub in fileList:
			video_length = 0
			lines = list(sub)
			while (video_length < (len(lines))):
				line = lines[video_length]
				imgPath = line.strip().split(' ')[0]
				find_str = os.path.dirname(imgPath)
				new_video_length = 0
				for line in lines:
					if find_str in line:
						new_video_length = new_video_length + 1
				videos.append(lines[video_length:video_length + new_video_length])
				video_length = video_length + new_video_length
		return videos


def default_list_train_val_reader(label_path, fileslist, num_subjects, data_domain):
		videos = []
		video_length = 0
		for filelist in fileslist:
				with open(label_path + filelist, 'r') as file:
						lines = list(file)
						while (video_length < len(lines)):
								line = lines[video_length]
								imgPath = line.strip().split(' ')[0]
								if (data_domain == "source"):
										find_str = os.path.dirname(imgPath)
								else:   
										find_str = os.path.dirname(os.path.dirname(imgPath))
								new_video_length = 0
								for line in lines:
										if find_str in line:
												new_video_length = new_video_length + 1
										else:   
												break
								videos.append(lines[video_length:video_length + new_video_length])
								lines = lines[video_length + new_video_length :]
		trainvideos = videos[0:num_subjects]
		valvideos = videos[num_subjects:len(videos)]
		return trainvideos, valvideos



targetlabel_path = "../Datasets/UNBC-McMaster/sublist/16/"
label_files = ["list_train_resample.txt"]
num_subjects = 15
data_domain = "target"

target_trainlist, target_vallist = default_list_train_val_reader(targetlabel_path,label_files, num_subjects,"target")

train_videos = default_list_reader(target_trainlist)
train_sequences, labels = default_seq_reader(train_videos, 64, 8)
print(len(target_trainlist))
print(len(train_videos))
print(len(train_sequences))
print(len(labels))



softtarget = torch.zeros(6, 6)
softtarget[0, :] = torch.FloatTensor([0.5, 0.3, 0.1, 0.05, 0.025, 0.025])
softtarget[1, :] = torch.FloatTensor([0.2, 0.5, 0.2, 0.0375, 0.0375, 0.025])
softtarget[2, :] = torch.FloatTensor([0.0375, 0.2, 0.5, 0.2, 0.0375, 0.025])
softtarget[3, :] = torch.FloatTensor([0.025, 0.0375, 0.2, 0.5, 0.2, 0.0375])
softtarget[4, :] = torch.FloatTensor([0.025, 0.0375, 0.0375, 0.2, 0.5, 0.2])
softtarget[5, :] = torch.FloatTensor([0.025, 0.025, 0.05, 0.1, 0.3, 0.5])

labels = np.asarray(labels)
softlabel = torch.zeros(2000, 6)
print(labels.shape[0])

k = 0
for i in range(2000):
	if (k>5):
		k = 0
	softlabel[i,:] = softtarget[k,:]
	k = k +1

print(softlabel.shape)
model = SLCD()


torch_dataset = Data.TensorDataset(softlabel)
loader = Data.DataLoader(
				dataset=torch_dataset, 
				batch_size=8, 
				shuffle=True, num_workers=2,)



net = torch.nn.Sequential(
		torch.nn.Linear(6, 6),
		torch.nn.LeakyReLU(),
		torch.nn.Linear(6, 6),
	)

optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum = 0.9)


def train(loader, model, optimizer):
#for epoch in range(100):
	model.train()
	for step, (batch) in enumerate(loader):
		optimizer.zero_grad()
		data_lab = Variable(batch[0])
		classes = torch.FloatTensor([0,1,2,3,4,5])
		data_outputs = torch.max(data_lab, dim=1)[1]
		outputs = model(data_lab)

		diff = []
		for i in range(batch[0].shape[0]):
			diff.append(torch.abs(classes - data_outputs[i]))
		
		LD = torch.stack(diff)

		CND = []
		for i in range(batch[0].shape[0]):
			if (data_outputs[i] == 0):
				CD = torch.abs(outputs[i][data_outputs[i]] - outputs[i][data_outputs[i]+1])
				CD = CD*LD[i]
			elif((data_outputs[i] == 5)):
				CD = torch.abs(outputs[i][data_outputs[i]] - outputs[i][data_outputs[i]-1])
				CD = CD*LD[i]
			else:
				CD = torch.abs(outputs[i][data_outputs[i]] - ((outputs[i][data_outputs[i]-1] + outputs[i][data_outputs[i]+1])/2))
				CD = CD*LD[i]				
			CND.append(CD)
		CND = torch.stack(CND)
		loss = torch.abs(outputs - CND)
		loss = torch.sum(loss)

		loss.backward()
		optimizer.step()
	print(loss)
	return model, loss



for epoch in range(250):
	model, loss = train(loader, model, optimizer)

for step, (batch) in enumerate(loader):	
	print(batch[0])
	outputs = model(batch[0])
	print(outputs)
	sys.exit()
