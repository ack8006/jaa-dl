from __future__ import print_function, division

import pickle

import torch 
import numpy as np

from convnet_leaky_relu import ConvNet

def predict(m, x_val):
    m.eval()
    x = Variable(x_val, requires_grad=False)
    output = m.forward(x)
    return output.data.numpy()

def helper_2(test_data, model_path, file_1, weight_1, file_2, weight_2):
	model = torch.load(model_path+file_1)
    predictions = predict(model, test_data) * weight_1

    model = torch.load(MODEL_PATH+model_2)
    predictions += predict(model, test_data) * model_2

    return predictions


if __name__ = '__main__':
	main()