import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import os.path as osp
import torch.nn.init as init
from EvaluationMetrics.ICC import compute_icc
from torch.autograd import Variable

from numpy import linalg as LA
#from tensorboardX import SummaryWriter
from utils.exp_utils import plot_features

#from visdom import Visdom
import utils.utils_progress
import matplotlib.pyplot as plt
import numpy as np
import logging
import sys
import utils.exp_utils as exp_utils

from torchsummary import summary
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import torch.nn.functional as F

#global plotter
#plotter = exp_utils.VisdomLinePlotter(env_name='praveen_Plots',port=8051)
#logging.basicConfig(filename=path + '/UNBC_source_recola_target_UNBC.log', level=logging.INFO)
#vis = Visdom()

def validate(val_loader, model, criterion, epoch, subject, TargetModeofSup):
	# switch to evaluate mode
	model.eval()
	PrivateTest_loss = 0
	running_val_loss = 0
	total = 0

	tar, out = [], []
	val_MAE = 0
	#all_features, all_labels = [], []

	for _, (input, target) in enumerate(val_loader):
		input = input.squeeze(1)
		with torch.no_grad():
			inputs = input.cuda()
			inputs = Variable(inputs)
			targets = target.type(torch.FloatTensor).cuda()
			model_targets = Variable(targets)
			model_outputs, _ = model(inputs, 0)

			## Frame-Level Prediction
			#t = inputs.size(2)
			#model_outputs = F.interpolate(outputs, t, mode='linear')
			model_outputs = model_outputs.squeeze(1).squeeze(2).squeeze(2)

		if (TargetModeofSup == 1):  ## Full Supervision
			model_targets = model_targets.view(-1, model_targets.shape[0]*model_targets.shape[1])
			model_outputs = model_outputs.view(-1, model_targets.shape[0]*model_targets.shape[1])
		else:    ## Weak Supervision
			model_outputs = torch.max(model_outputs, dim=1)[0]
			model_targets = torch.max(model_targets, dim=1)[0]

			model_outputs = model_outputs.view(-1, model_outputs.shape[0])#.squeeze()
			model_targets = model_targets.view(-1, model_targets.shape[0])#.squeeze()

		loss = criterion(model_outputs, model_targets)
		PrivateTest_loss += loss.item()
		total += targets.size(0)
		running_val_loss += loss.item() * targets.size(0)

		#all_features.append(features.data.cpu().numpy())
		#all_labels.append(targets.data.cpu().numpy())

		#out.append(outputs.data.cpu().numpy())
		#tar.append(targets.data.cpu().numpy())
		out = np.concatenate([out, model_outputs.squeeze(0).detach().cpu().numpy()])
		tar = np.concatenate([tar, model_targets.squeeze(0).detach().cpu().numpy()])

	PrivateTest_loss = PrivateTest_loss / total

	#all_features = np.concatenate(all_features, 0)
	#all_labels = np.concatenate(all_labels, 0)

	#plot_features(all_features, all_features, all_labels, 6, epoch, dname2, prefix='val', subject=subject)
	val_MAE = mean_absolute_error(tar, out)
	val_MSE = mean_squared_error(tar, out)
	print("MAE : " + str(val_MAE))
	# print(val_MAE)
	print("MSE : " + str(val_MSE))
	# print(val_MSE)
	#print("Accuracy : " + str(Val_acc))
	# print(Val_acc)

	pearson_measure, _ = pearsonr(out, tar)
	print("PCC :" + str(pearson_measure))
	# print(pearson_measure)
	logging.info("Val Loss: " + str(running_val_loss / total))
	logging.info("Val_accuracy: " + str(pearson_measure))
	logging.info("MAE : " + str(val_MAE))
	# print((running_val_loss / total), Val_acc)
	val_icc = compute_icc(out, tar)

	print("ICC : " + str(val_icc))
	logging.info("ICC : " + str(val_icc))
	#plotter.plot('loss', 'val', 'Class Loss', epoch, (running_val_loss / total))
	#plotter.plot('acc', 'val', 'Class Accuracy', epoch, Val_acc.tolist())

	return (running_val_loss / total), pearson_measure, val_MAE, val_icc

