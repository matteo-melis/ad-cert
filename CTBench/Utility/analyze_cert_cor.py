import torch
import sys
sys.path.append("../")
from utils import load_perf_from_json
import os
import numpy as np
import matplotlib.pyplot as plt
from loaders import get_mnist
from math import comb

root = "../../CTBenchRelease"
dataset = "cifar10"
eps = "2.255"
method = ("IBP", "TAPS", "STAPS", "SABR", "MTL-IBP", "CROWN-IBP")
device = "cuda" if torch.cuda.is_available() else "cpu"

nat = []
cert = []
for m in method:
    d = load_perf_from_json(os.path.join(root, dataset, eps, m), "complete_cert.json")["is_nat_cert_accurate"]
    is_nat_accu = np.array([int(i[0]) for i in d])
    is_cert_accu = np.array([int(i[1]) for i in d])
    assert (is_nat_accu >= is_cert_accu).all()
    nat.append(is_nat_accu)
    cert.append(is_cert_accu)
is_nat_accu = np.stack(nat, axis=1)
is_cert_accu = np.stack(cert, axis=1)

# some samples are not certified by all methods
nat_freq = np.sum(is_nat_accu, axis=1)
cert_freq = np.sum(is_cert_accu, axis=1)
freq, cert_counts = np.unique(cert_freq, return_counts=True)
print(cert_counts.tolist())

# # visualize samples with different cert difficulty
# hard_idx = np.where((cert_freq == 0) & (nat_freq>=4))[0]
# # print(hard_idx)
# train_set, test_set, input_size, input_channels, n_class = get_mnist()
# for idx in hard_idx:
#     plt.imshow(test_set[idx][0][0])
#     plt.savefig(f"temp_figures/hard{test_set[idx][1]}_{idx}.png")

# expectation given i.i.d. assumption
def expected_count(probs, n:int):
    '''
    assume independent events, the probability of a count variable that counts the total number of failed events
    '''
    k = len(probs)
    assert k == 6
    # k = 0
    p0 = np.prod([1-p for p in probs])
    # k = 1
    temp = [probs[i]*p0/(1-probs[i]) for i in range(k)]
    p1 = np.sum(temp)
    # k = 2
    temp = [probs[i]*probs[j]*p0/(1-probs[i])/(1-probs[j]) for i in range(k) for j in range(i+1, k)]
    p2 = np.sum(temp)
    # k = 3
    temp = [probs[i]*probs[j]*probs[l]*p0/(1-probs[i])/(1-probs[j])/(1-probs[l]) for i in range(k) for j in range(i+1, k) for l in range(j+1, k)]
    p3 = np.sum(temp)
    # k = 6
    p6 = np.prod(probs)
    # k = 5
    temp = [(1-probs[i])*p6/(probs[i]) for i in range(k)]
    p5 = np.sum(temp)
    # k = 4
    temp = [(1-probs[i])*(1-probs[j])*p6/(probs[i])/(probs[j]) for i in range(k) for j in range(i+1, k)]
    p4 = np.sum(temp)
    results = np.array([p0, p1, p2, p3, p4, p5, p6])
    return results * n

n = 10000
cert_accu = np.sum(is_cert_accu, axis=0) / n
print(cert_accu)
result = np.round(expected_count(cert_accu, n), 0)
print([int(i) for i in result])
print(np.sum(result))