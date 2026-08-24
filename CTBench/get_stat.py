import torch
import torch.nn as nn
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from args_factory import get_args
from loaders import get_loaders
from utils import Scheduler, Statistics
from networks import get_network, fuse_BN_wrt_Flatten
from model_wrapper import BoxModelWrapper, BasicModelWrapper, get_model_wrapper
import os
from utils import write_perf_to_json, load_perf_from_json, seed_everything
from tqdm import tqdm
import random
import numpy as np
from regularization import compute_fast_reg, compute_vol_reg, compute_L1_reg, compute_PI_reg
import time
from datetime import datetime
from AIDomains.abstract_layers import Sequential, Flatten, Linear, ReLU, Conv2d, _BatchNorm
from AIDomains.zonotope import HybridZonotope
from AIDomains.ai_util import construct_C
import matplotlib.pyplot as plt
from Utility.PI_functions import compute_tightness

import warnings
warnings.filterwarnings("ignore")

def test_loop(model_wrapper:BasicModelWrapper, eps, test_loader, device, args):
    model_wrapper.net.eval()

    if hasattr(model_wrapper, "num_steps"):
        # PGD
        model_wrapper.num_steps = args.test_steps
    elif hasattr(model_wrapper, "latent_search_steps"):
        # TAPS and STAPS
        model_wrapper.latent_search_steps = args.test_steps
    elif hasattr(model_wrapper, "input_search_steps"):
        # SABR (note STAPS also has input_search_steps but will not be updated here)
        model_wrapper.input_search_steps = args.test_steps

    model_wrapper.store_box_bounds = False
    model_wrapper.summary_accu_stat = True
    model_wrapper.freeze_BN = True
    nat_accu_stat, robust_accu_stat, loss_stat = Statistics.get_statistics(3)

    pbar = tqdm(test_loader)
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(pbar):
            x, y = x.to(device), y.to(device)
            (loss, nat_loss, cert_loss), (nat_accu, robust_accu) = model_wrapper.compute_model_stat(x, y, eps) # already called eval, so do not need to close BN again in the common step.
            nat_accu_stat.update(nat_accu, len(x))
            robust_accu_stat.update(robust_accu, len(x))

            # if args.L1_reg > 0:
            #     loss = loss + args.L1_reg * compute_L1_reg(model_wrapper.net)
            # if args.PI_reg > 0:
            #     PI_reg = args.PI_reg * compute_PI_reg(model_wrapper.net, x, y, 1e-6, args.num_classes, relu_adjust="local")
            #     loss = loss + PI_reg
            loss_stat.update(loss.item(), len(x))

            pbar.set_postfix_str(f"nat_accu: {nat_accu_stat.avg:.3f}, robust_accu: {robust_accu_stat.avg:.3f}, test_loss: {loss_stat.avg:.3f}")

    return nat_accu_stat.avg, robust_accu_stat.avg, loss_stat.avg


def PI_loop(net, eps, test_loader, device, num_classes, args, relu_adjust="local", max_examined_class:int=8):
    assert relu_adjust in ["local"], "Only local relu adjustment is supported for now."
    net.eval()
    BN_layers = [layer for layer in net if isinstance(layer, _BatchNorm)]
    for layer in BN_layers:
        layer.set_current_to_running() # Essential for testing; compute_tightness will use current stat for computation

    PI_stat = Statistics()

    pbar = tqdm(test_loader)
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(pbar):
            x, y = x.to(device), y.to(device)
            tightness = compute_tightness(net, x, y, eps, only_W=False, num_classes=num_classes, relu_adjust=relu_adjust, error_check=False, verbose=False, max_examined_class=max_examined_class)
            PI_stat.update(tightness.mean().item(), len(x))
            pbar.set_postfix_str(f"PI: {PI_stat.avg:.3E}")
    net.reset_bounds()
    return PI_stat.avg

def relu_loop(net, eps, test_loader, device, args):
    net.eval()
    BN_layers = [layer for layer in net if isinstance(layer, _BatchNorm)]
    relu_layers = [layer for layer in net if isinstance(layer, ReLU)]

    original_stat = [layer.update_stat for layer in BN_layers]
    for layer in BN_layers:
        layer.update_stat = False

    dead, unstable, active = Statistics.get_statistics(3)
    pbar = tqdm(test_loader)
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(pbar):
            x = x.to(device)
            num_dead, num_active, num_total = 0, 0, 0
            net.reset_bounds()
            abs_input = HybridZonotope.construct_from_noise(x, eps, "box")
            abs_out = net(abs_input)
            for layer in relu_layers:
                lb, ub = layer.bounds
                num_total += lb.numel()
                num_dead += (ub < 0).sum().item()
                num_active += (lb > 0).sum().item()
            num_unstable = num_total - num_dead - num_active
            dead.update(num_dead/num_total, len(x))
            unstable.update(num_unstable/num_total, len(x))
            active.update(num_active/num_total, len(x))
            pbar.set_postfix_str(f"dead: {dead.avg:.3f}; unstable: {unstable.avg:.3f}; active: {active.avg:.3f}")
    
    for layer, stat in zip(BN_layers, original_stat):
        layer.update_stat = stat
    net.reset_bounds()
    return dead.avg, unstable.avg, active.avg

def BoxSize_Loop(net, eps, test_loader, device, num_class:int, args):
    net.eval()
    bs_stat = Statistics()
    net.reset_bounds()
    pbar = tqdm(test_loader)
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(pbar):
            x, y = x.to(device), y.to(device)
            abs_input = HybridZonotope.construct_from_noise(x, eps, "box")
            C = construct_C(num_class, y)
            abs_out = net(abs_input, C=C)
            lb, ub = abs_out.concretize()
            bs = ((ub - lb) / 2).mean()
            bs_stat.update(bs.item(), len(x))
            net.reset_bounds()
            pbar.set_postfix_str(f"Box_size: {bs_stat.avg:.3E}")
    return bs_stat.avg

def Margin_Loop(net, test_loader, device, num_class:int, args):
    # Computes the margin for the natural inputs. Margin is defined as largest logit minus the second largest logit
    net.eval()
    margin_stat = Statistics()
    net.reset_bounds()
    pbar = tqdm(test_loader)
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(pbar):
            x, y = x.to(device), y.to(device)
            out = net(x)
            top2, _ = torch.topk(out, k=2, dim=1)
            margin = (top2[:, 0] - top2[:, 1]).abs().mean()
            margin_stat.update(margin.item(), len(x))
            net.reset_bounds()
            pbar.set_postfix_str(f"Margin: {margin_stat.avg:.3E}")
    return margin_stat.avg

def run_PI(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    perf_dict = {'vanilla_PI_curve':[], 'local_PI_curve':[], 'shrink_PI_curve':[]}
    verbose = False

    loaders, input_size, input_channel, n_class = get_loaders(args)
    input_dim = (input_channel, input_size, input_size)
    if len(loaders) == 3:
        train_loader, val_loader, test_loader = loaders
    else:
        train_loader, test_loader = loaders
        val_loader = None

    net = get_network(args.net, args.dataset, device, init=args.init)
    net = Sequential.from_concrete_network(net, input_dim, disconnect=True)
    print(net)

    net.load_state_dict(torch.load(args.load_model))
    local = PI_loop(net, args.test_eps, test_loader, device, n_class, args, relu_adjust="local")
    perf_dict[f"final_local_PI"] = f"{local:.1e}"

    perf_dict["time"] = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
    write_perf_to_json(perf_dict, os.path.dirname(args.load_model), "PI.json")
    
def run_BoxSize(args, normalize:bool=True):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    perf_dict = {}
    verbose = False

    loaders, input_size, input_channel, n_class = get_loaders(args)
    input_dim = (input_channel, input_size, input_size)
    if len(loaders) == 3:
        train_loader, val_loader, test_loader = loaders
    else:
        train_loader, test_loader = loaders
        val_loader = None

    net = get_network(args.net, args.dataset, device, init=args.init)
    net = Sequential.from_concrete_network(net, input_dim, disconnect=True)
    print(net)

    net.load_state_dict(torch.load(args.load_model))

    # vanilla, local, shirnk = PI_loop(net, args.test_eps, test_loader, device, n_class, args)
    margin = Margin_Loop(net, test_loader, device, n_class, args)
    bs = BoxSize_Loop(net, args.test_eps, test_loader, device, n_class, args)
    perf_dict[f"final_Boxsize"] = round(bs, 4)
    perf_dict[f"final_margin"] = round(margin, 4)
    perf_dict[f"Normalized_BS"] = round(bs / margin, 4)

    perf_dict["time"] = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
    write_perf_to_json(perf_dict, os.path.dirname(args.load_model), "Boxsize.json")
    
def run_relu(args, eps:float=0):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    perf_dict = {}
    verbose = False

    loaders, input_size, input_channel, n_class = get_loaders(args)
    input_dim = (input_channel, input_size, input_size)
    if len(loaders) == 3:
        train_loader, val_loader, test_loader = loaders
    else:
        train_loader, test_loader = loaders
        val_loader = None

    net = get_network(args.net, args.dataset, device, init=args.init)
    net = Sequential.from_concrete_network(net, input_dim, disconnect=True)
    print(net)

    net.load_state_dict(torch.load(args.load_model))
    net.eval()

    dead, unstable, active = relu_loop(net, eps, test_loader, device, args)
    perf_dict[f"dead_relu"] = round(dead * 100, 2)
    perf_dict[f"unstable_relu"] = round(unstable * 100, 2)
    perf_dict[f"active_relu"] = round(active * 100, 2)

    perf_dict["time"] = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
    if eps > 0:
        write_perf_to_json(perf_dict, os.path.dirname(args.load_model), "relu.json")
    else:
        write_perf_to_json(perf_dict, os.path.dirname(args.load_model), "relu_0.json")

def run_unstable_relu_lower_bound(args, eps, restarts=5):
    '''
    feed concrete inputs sampled from the sepcification, thus get the lower bound of the ratio of unstable relus
    '''
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    perf_dict = {}
    verbose = False

    loaders, input_size, input_channel, n_class = get_loaders(args)
    input_dim = (input_channel, input_size, input_size)
    if len(loaders) == 3:
        train_loader, val_loader, test_loader = loaders
    else:
        train_loader, test_loader = loaders
        val_loader = None

    net = get_network(args.net, args.dataset, device, init=args.init)
    net = Sequential.from_concrete_network(net, input_dim, disconnect=True)

    net.load_state_dict(torch.load(args.load_model))
    net.eval()

    relu_layers = [layer for layer in net if isinstance(layer, ReLU)]

    unstable_stat = Statistics()
    pbar = tqdm(test_loader)
    for orig_x, orig_y in pbar:
        orig_x = orig_x.to(device)
        perturb_range = ((orig_x - eps).clamp(min=0), (orig_x + eps).clamp(max=1))
        center = (perturb_range[0] + perturb_range[1]) / 2
        radius = (perturb_range[1] - perturb_range[0]) / 2
        dead = [0]*len(relu_layers)
        active = [0]*len(relu_layers)
        for i in range(restarts):
            # sample around the clean input gets higher rates
            x = (orig_x + eps * torch.randn_like(orig_x).clamp(-1, 1)).clamp(0,1)
            net.reset_bounds()
            abs_input = HybridZonotope.construct_from_noise(x, 0, "box")
            abs_out = net(abs_input)
            for i, layer in enumerate(relu_layers):
                activation = layer.bounds[0].flatten()
                dead[i] = dead[i] | (activation < 0)
                active[i] = active[i] | (activation > 0)
        unstable = [d & a for d, a in zip(dead, active)]
        unstable = torch.concat(unstable, dim=0).float().mean().item()
        unstable_stat.update(unstable*100, len(orig_x))
        pbar.set_postfix_str(f"unstable: {unstable_stat.avg:.3f}")
    perf_dict[f"unstable_relu_lower_bound"] = round(unstable_stat.avg, 2)
    write_perf_to_json(perf_dict, os.path.dirname(args.load_model), f"unstable_relu_lower_bound_{eps}.json")


def run_train_accu(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    perf_dict = {}
    verbose = False

    loaders, input_size, input_channel, n_class = get_loaders(args)
    input_dim = (input_channel, input_size, input_size)
    if len(loaders) == 3:
        train_loader, val_loader, test_loader = loaders
    else:
        train_loader, test_loader = loaders
        val_loader = None

    net = get_network(args.net, args.dataset, device, init=args.init)
    net = Sequential.from_concrete_network(net, input_dim, disconnect=True)
    print(net)

    net.load_state_dict(torch.load(args.load_model))

    model_wrapper = get_model_wrapper(args, net, device, input_dim)
    model_wrapper.max_eps = args.train_eps
    model_wrapper.robust_weight = 1
    model_wrapper.net.eval()

    nat_accu, cert_accu, loss = test_loop(model_wrapper, args.train_eps, train_loader, device, args)
    perf_dict["train_nat_accu"] = round(nat_accu, 4)
    perf_dict["train_cert_accu"] = round(cert_accu, 4)
    perf_dict["train_loss"] = round(loss, 4)

    perf_dict["time"] = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
    write_perf_to_json(perf_dict, os.path.dirname(args.load_model), "accu.json")

def run_param_sign(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    perf_dict = {}

    loaders, input_size, input_channel, n_class = get_loaders(args)
    input_dim = (input_channel, input_size, input_size)

    net = get_network(args.net, args.dataset, device, init=args.init)
    net = Sequential.from_concrete_network(net, input_dim, disconnect=True)

    net.load_state_dict(torch.load(args.load_model))

    for name, param in net.named_parameters():
        perf_dict[name] = (round((param > 0).sum().item() / param.numel(), 4), round((param < 0).sum().item() / param.numel(), 4), round((param == 0).sum().item() / param.numel(), 4))
    write_perf_to_json(perf_dict, os.path.dirname(args.load_model), "param_sign.json")


def run_mnist_corrupted(args):
    assert args.dataset == "mnist", "Only MNIST models can be evaluated on MNIST-C."
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    perf_dict = {}

    input_dim = (1, 28, 28)
    net = get_network(args.net, "mnist", device, init=args.init)
    net = Sequential.from_concrete_network(net, input_dim, disconnect=True)
    net.load_state_dict(torch.load(args.load_model))
    net.eval()
    # get clean accuracy to adjust for the performance drop
    test_loader = get_loaders(args, shuffle_test=False)[0][-1]
    accu_stat = Statistics()
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        out = net(x)
        nat_accu = (out.argmax(1) == y).float().mean().item()
        accu_stat.update(nat_accu, len(x))
    clean_accu = accu_stat.avg

    corruptions = ["brightness", "canny_edges", "dotted_line", "fog", "glass_blur", "impulse_noise", "motion_blur", "rotate", "scale", "shear", "shot_noise", "spatter", "stripe", "translate", "zigzag"]

    print("Testing MNIST-C")
    for corruption in corruptions:
        data_path = os.path.join("data", "mnist_c", corruption)
        x = np.load(os.path.join(data_path, "test_images.npy")) / 255
        y = np.load(os.path.join(data_path, "test_labels.npy"))
        x = np.transpose(x, (0, 3, 1, 2)).astype(np.float32)
        
        dataset = torch.utils.data.TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.test_batch, shuffle=False, num_workers=8)
        accu_stat = Statistics()
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            out = net(x)
            accu = (out.argmax(1) == y).float().mean().item()
            accu_stat.update(accu, len(x))
        accu = accu_stat.avg
        perf_dict[corruption] = round(accu / clean_accu, 4)
    mean_accu = np.mean(list(perf_dict.values()))
    perf_dict["mean"] = round(mean_accu, 4)
    # distinguish brightness
    # perf_dict["no_brightness_mean"] = round((np.sum(list(perf_dict.values())) - perf_dict["brightness"]) / (len(perf_dict) - 1), 4)
    print(perf_dict)
    write_perf_to_json(perf_dict, os.path.dirname(args.load_model), "mnist_c.json")

def run_cifar10_corrupted(args):
    assert args.dataset == "cifar10", "Only CIFAR-10 models can be evaluated on CIFAR-10-C."
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    perf_dict = {}

    input_dim = (3, 32, 32)
    net = get_network(args.net, "cifar10", device, init=args.init)
    net = Sequential.from_concrete_network(net, input_dim, disconnect=True)
    net.load_state_dict(torch.load(args.load_model))
    net.eval()
    # get clean accuracy to adjust for the performance drop
    test_loader = get_loaders(args, shuffle_test=False)[0][-1]
    accu_stat = Statistics()
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        out = net(x)
        nat_accu = (out.argmax(1) == y).float().mean().item()
        accu_stat.update(nat_accu, len(x))
    clean_accu = accu_stat.avg


    corruptions = ["brightness", "contrast", "defocus_blur", "elastic_transform", "fog", "frost", "gaussian_blur", "gaussian_noise", "glass_blur", "impulse_noise", "jpeg_compression", "motion_blur", "pixelate", "saturate", "shot_noise", "spatter", "speckle_noise", "zoom_blur"]

    print("Testing CIFAR-10-C")
    labels = np.load(os.path.join("data", "CIFAR-10-C", "labels.npy"))
    for corruption in corruptions:
        data_path = os.path.join("data", "CIFAR-10-C")
        data = np.load(os.path.join(data_path, f"{corruption}.npy"))
        x = data / 255.
        x = np.transpose(x, (0, 3, 1, 2)).astype(np.float32)
        y = labels
    
        dataset = torch.utils.data.TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.test_batch, shuffle=False, num_workers=8)
        accu_stat = Statistics()
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            out = net(x)
            accu = (out.argmax(1) == y).float().mean().item()
            accu_stat.update(accu, len(x))
        accu = accu_stat.avg
        perf_dict[corruption] = round(accu / clean_accu, 4)
    mean_maintainance = np.mean(list(perf_dict.values()))
    perf_dict["mean"] = round(mean_maintainance, 4)
    print(perf_dict)
    write_perf_to_json(perf_dict, os.path.dirname(args.load_model), "cifar10_c.json")

def run_tinyimagenet_corrupted(args):
    assert args.dataset == "tinyimagenet", "Only TinyImageNet models can be evaluated on TinyImageNet-C."
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    perf_dict = {}

    input_dim = (3, 64, 64)
    net = get_network(args.net, "tinyimagenet", device, init=args.init)
    net = Sequential.from_concrete_network(net, input_dim, disconnect=True)
    net.load_state_dict(torch.load(args.load_model))
    net.eval()
    # get clean accuracy to adjust for the performance drop
    test_loader = get_loaders(args, shuffle_test=False)[0][-1]
    accu_stat = Statistics()
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        out = net(x)
        nat_accu = (out.argmax(1) == y).float().mean().item()
        accu_stat.update(nat_accu, len(x))
    clean_accu = accu_stat.avg

    corruptions = ["brightness", "contrast", "defocus_blur", "elastic_transform", "fog", "frost", "gaussian_noise", "glass_blur", "impulse_noise", "jpeg_compression", "motion_blur", "pixelate", "shot_noise", "snow", "zoom_blur"]
    severity = 1

    print("Testing TinyImageNet-C")
    for corruption in corruptions:
        data_path = os.path.join("data", "Tiny-ImageNet-C")
        dataset = ImageFolder(os.path.join(data_path, corruption, str(severity)), transform=transforms.ToTensor())
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.test_batch, shuffle=False, num_workers=8)
        accu_stat = Statistics()
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            out = net(x)
            accu = (out.argmax(1) == y).float().mean().item()
            accu_stat.update(accu, len(x))
        perf_dict[corruption] = round(accu_stat.avg / clean_accu, 4)
    mean_maintainance = np.mean(list(perf_dict.values()))
    perf_dict["mean"] = round(mean_maintainance, 4)
    print(perf_dict)
    write_perf_to_json(perf_dict, os.path.dirname(args.load_model), "tinyimagenet_c.json")

def main():
    args = get_args(include=["basic", "train"])
    seed_everything(args.random_seed)
    run_PI(args)
    # # # # # run_BoxSize(args)
    run_relu(args, eps=args.test_eps)
    run_relu(args, eps=0)
    # # # run_train_accu(args)
    # # run_mnist_corrupted(args)
    run_cifar10_corrupted(args)
    # run_tinyimagenet_corrupted(args)
    run_unstable_relu_lower_bound(args, eps=args.test_eps, restarts=50)

if __name__ == '__main__':
    main()
