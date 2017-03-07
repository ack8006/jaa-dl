import torch
import pickle
import numpy as np
# from convnet_leaky_relu import ConvNet
# from convnet_heavy_dropout import ConvNet
# from convnet3_deep3 import ConvNet
from convnet3_deep2 import ConvNet
from torch.autograd import Variable


# model = torch.load("saved_models_leaky_relu/best_cnn_d14d25b32e766acc9929.model")
# model = torch.load("saved_models3/best_heavy_dropout_d2525252555b16e195acc9925.model")
# model = torch.load("saved_models3/best_cnn_d12d25b16e118acc9932.model")
model = torch.load("saved_models2/best_cnn_d4b16e73acc9929.model")

model.eval()
test_old = pickle.load(open("data/test_old.p"))
test_data = test_old.test_data.numpy()
test_data = np.array([(x.flatten()/255.0).reshape(1,28,28) for x in test_data])
test_data = torch.from_numpy(test_data).float().resize_(len(test_data),1,28,28)
test_labels = test_old.test_labels

def predict(m, x_val):
    x = Variable(x_val, requires_grad=False)
    output = m.forward(x)
    return output.data.numpy()

pred_y = predict(model, test_data)

with open("preds_conv3_deep2_sun_best_test_old.p", "wb") as f:
    pickle.dump(pred_y, f)


# print(100. * np.mean(pred_y == test_labels.numpy()))

# with open("test_final_predictions_latest.p", "wb") as f:
#     pickle.dump(pred_y, f)
