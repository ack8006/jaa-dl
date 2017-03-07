from __future__ import print_function, division

import torch 
from torch.autograd import Variable
import pickle

from convnet_leaky_relu import ConvNet

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

def helper_2(test_data, model_path, file_1, weight_1, file_2, weight_2):
    model = torch.load(model_path+file_1)
    predictions = predict(model, test_data) * weight_1

    model = torch.load(model_path+file_2)
    predictions += predict(model, test_data) * weight_2

    return predictions

if __name__ == '__main__':
    test_data, _ = load_valid_data()
    pred = helper_2(test_data, 'final_model_save/pre-gen-models/',
                        'convleaky_mdl2.model', 0.15789473684210525,
                        'convleaky_mdl1.model', 0.15789473684210525)
    pickle.dump(pred, open('helper_2_dump.p', 'w'))