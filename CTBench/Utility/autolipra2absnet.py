'''
Convert a AutoLipra model to Abstract Network
'''
import torch
import torch.nn as nn
import os
import sys
sys.path.append("../")
from networks import get_network
from AIDomains.abstract_layers import Sequential

root = "../../../expressive-losses/model_tinyimagenet/cnn_7layer_bn_imagenet_fast_1713269014_ckpt_last"
save_path = "../model_test/cnn_7layer_bn_tinyimagenet.ckpt"
input_dim = (3, 64, 64)

model = torch.load(root, map_location=torch.device('cpu'))["state_dict"]
print(model.keys())

# make sure to get a network with the same architecture
net = get_network("cnn_7layer_bn_tinyimagenet", dataset="tinyimagenet", device="cpu")
state_dict = net.state_dict()

# main conversion
for k in state_dict.keys():
    model_key_id = int(k.split(".")[0])-1 # normalization layer offset
    model_key = f"{model_key_id}." + ".".join(k.split(".")[1:])
    if model_key in model.keys():
        state_dict[k] = model[model_key]
    else:
        print("key not found: ", k)

net.load_state_dict(state_dict)
net = Sequential.from_concrete_network(net, input_dim, disconnect=False)

os.makedirs(os.path.dirname(save_path), exist_ok=True)
torch.save(net.state_dict(), save_path)
print("Model saved to ", save_path)