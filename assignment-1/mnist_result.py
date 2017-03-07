from __future__ import print_function, division

import pickle

import torch 
import numpy as np
import pandas as pd


#MODEL_PATH = 'final_model_save/'
MODEL_PATH = 'final_model_save/pre-gen-models/'

FINAL_WEIGHTS = [0.15789473684210525, 0.2368421052631579, 0.15789473684210525,
                 0.15789473684210525, 0.2105263157894737, 0.07894736842105263]

def load_valid_data():
    valid_data = pickle.load(open('data/generated_valid_data_norm.p', 'rb'))
    valid_data = torch.from_numpy(valid_data).float().resize_(len(valid_data),1,28,28)

    valid_label = pickle.load(open('data/generated_valid_labels.p', 'rb'))
    valid_label = torch.from_numpy(valid_label).long()
    return valid_data, valid_label

def load_test_data():
    test_data = pickle.load(open("data/test.p", "rb"))
    test_data = test_data.test_data.numpy()
    test_data = np.array([(x.flatten()/255.0).reshape(1,28,28) for x in test_data])
    test_data = torch.from_numpy(test_data).float().resize_(len(test_data),1,28,28)
    return test_data

def predict(m, x_val):
    m.eval()
    x = Variable(x_val, requires_grad=False)
    output = m.forward(x)
    return output.data.numpy()

def main():
    test_data , valid_label = load_valid_data()
    #test_data = load_test_data()


    print(1)
    from convnet3_deep3 import ConvNet
    model = torch.load(MODEL_PATH+'conv3deep3_mdl1.model')
    predictions = predict(model, test_data) * FINAL_WEIGHTS[0]

    print(2)
    model = torch.load(MODEL_PATH+'conv3deep3_mdl2.model')
    predictions += predict(model, test_data) * FINAL_WEIGHTS[1]

    print(3)
    from convnet_leaky_relu import ConvNet
    model = torch.load(MODEL_PATH+'convleaky_mdl2.model')
    predictions = predict(model, test_data) * FINAL_WEIGHTS[2]

    print(4)
    model = torch.load(MODEL_PATH+'convleaky_mdl1.model')
    predictions += predict(model, test_data) * FINAL_WEIGHTS[3]

    print(5)
    from convnet_heavy_dropout import ConvNet
    model = torch.load(MODEL_PATH+'convheavy_mdl1.model')
    predictions = predict(model, test_data) * FINAL_WEIGHTS[4]

    print(6)
    model = torch.load(MODEL_PATH+'convleaky_mdl1.model')
    predictions += predict(model, test_data) * FINAL_WEIGHTS[5]

    predictions = predictions.argmax(axis=1)
    print(100. * np.mean(predictions == valid_label.numpy()))


if __name__ == '__main__':
    main()
