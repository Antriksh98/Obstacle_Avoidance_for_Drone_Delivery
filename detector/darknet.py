from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import variable
import numpy as np

#cfgfile = '/home/antriksh/Documents/Antriksh/codes/avionics/detector/cf'g

def parse_cfg(cfgfile):
	"""
 	Takes a configuration file
 	Returns a list of blocks. Each blocks describes a block in the 
 	neural network to be built Block is represented as a dictionary 
 	in the list
 	"""	

	file = open(cfgfile, 'r')
	lines = file.read().split('/n') # stores lines in a list
	lines = [x for x in lines if len(x)>0] # gets rid of empty lines
	lines = [x for x in lines if x[0] != '#'] # gets rid of comments
	lines = [x.rstrip().lstrip() for x in lines] # gets rid of fringe whitespaces

	block = {}
	blocks = []

	for line in lines:
		if (line[0] == "["): # This marks the start of a new block
			if len(block) != 0: # If block is not empty, it is storing values from the previous blocks.
				blocks.append(block)# Add it to the blocks list
				block = {}# Re-initialise the block
			block["type"] = line[1:-1].rstrip()
		else:
			key, value = line.split("=")
			block[key.rstrip()] = value.lstrip()

	blocks.append(block)

	return blocks

class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()

class DetectionLayer(nn.Module):
	def __init__(self, anchors):
		super(DetectionLayer, self).__init__()
		self.anchors = anchors

def create_modules(blocks):

	net_info = blocks[0]
	module_list = nn.ModuleList()
	prev_filters = 3
	output_filters =[]

	module = nn.Sequential()#If does not work remove it.
	filters = 0 #If does not work remove it.

	for index, x in enumerate(blocks[1: ]):
		module = nn.Sequential()

		# Check the type of blocks
		#Create a new module for the blocks
		#append to module_list

		if (x["type"] == "convolutional"):
			#Get the info about the layer
			activation = x["activation"]
			try:
				batch_normalize = int(x["batch_normalize"])
				bias = False
			except:
				batch_normalize = 0
				bias = True

			filters = int(x["filters"])
			padding = int(x["pad"])
			stride = int(x["stride"])
			kernel_size = int(x["size"])

			if padding:
				pad = (kernel_size-1)//2
			else:
				pad = 0

			#Add the convolutional layer
			conv = nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias = bias)
			module.add_module("conv_{0}".format(index), conv)

			#Add the Batch Norm layer

			if batch_normalize:
				bn = nn.BatchNorm2d(filters)
				module.add_module("conv_{0}".format(index), bn)

			#Check the activation
			#It is either Linear or a leaky ReLU for YOLO
			if activation == "leaky":
				activn = nn.LeakyReLU(0.1, inplace = True)
				module.add_module("leaky_{0}".format(index), activn)
		
		#If it's upsampling layer
		#We use Bilinear2dUpsampling
		elif (x["type"] == "upsampling"):
			stride = int(x["stride"])
			upsample = nn.upsample(scale_factor = 2, mode = "bilinear")
			module.add_module("upsample_{}".format(index), upsample)
		
		#If it is a route layer

		elif (x["type"] == "route"):
			x["layers"] = x["layers"].split(',')
			#start of a route
			start = int(x["layers"][0])
			#end, if there exisis one.

			try:
				end = int(x["layers"][1])
			except:
				end = 0

			# Positive anotation
			if start > 0:
				start = start - index
			if end > 0:
				end = end - index
			route = EmptyLayer()
			module.add_module("route_{0}".format(index), route)
			if end < 0:
				filters = output_filters[index+start] + output_filters[index+end]
			else:
				filters = output_filters[index+start]

		#shortcut corresponds to skip connection
		elif x["type"] == "shortcut":
			shortcut = EmptyLayer()
			module.add_module("shortcut_{}".format(index), shortcut)

		#Yolo is the detection layer
		elif x["type"] == "yolo":
			mask = x["mask"].split(",")
			mask = [int(x) for x in mask]

			anchors = x["anchors"].split(",")
			anchors = [int(a) for a in anchors]
			anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors), 2)]
			anchors = [anchors[i] for i in mask]

			detection = DetectionLayer(anchors)
			module.add_module("Detection_{}".format(index), detection)

	module_list.append(module)
	prev_filters = filters	
	output_filters.append(filters)

	return (net_info, module_list)	

blocks = parse_cfg("cfg/yolov3.cfg")
print(create_modules(blocks))

#https://blog.paperspace.com/how-to-implement-a-yolo-v3-object-detector-from-scratch-in-pytorch-part-2/
#Use the link above for further assistance.
# check lines 151, 62, 78, 152. These are the lines producing the error.
#I have added line 58, 59 to solve the problem but this does not seem to work.
