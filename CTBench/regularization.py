import torch
import torch.nn.functional as F
from AIDomains.abstract_layers import Linear, Conv2d, ReLU, _BatchNorm
from AIDomains.zonotope import HybridZonotope
from Utility.PI_functions import compute_tightness
import numpy as np

def compute_L1_reg(abs_net):
    loss = 0
    for module in abs_net.modules():
        if isinstance(module, Linear) or isinstance(module, Conv2d):
            loss = loss + module.weight.abs().sum()
    return loss

def compute_L2_reg(abs_net):
    loss = 0
    for module in abs_net.modules():
        if isinstance(module, Linear) or isinstance(module, Conv2d):
            loss = loss + (module.weight**2).sum()
    return loss

def compute_neg_reg(abs_net, L1_ratio:float=0.5, L1_cutoff:float=0.05):
    loss = 0
    cnt = 0
    for module in abs_net.modules():
        if isinstance(module, Linear) or isinstance(module, Conv2d):
            neg_w = module.weight[module.weight < 0]
            # ref: https://arxiv.org/pdf/1802.00003.pdf
            L1 = neg_w[neg_w < -L1_cutoff].abs().sum() + (neg_w[neg_w >= -L1_cutoff]**2 + L1_cutoff**2).sum() / (2*L1_cutoff)
            L2 = 0.5 * (neg_w**2).sum()
            loss = loss + L1_ratio * L1 + (1 - L1_ratio) * L2
            cnt += 1
    loss /= (cnt + 1e-8)
    return loss

def compute_PI_reg(abs_net, x, y, num_classes:int, eps:float=None, relu_adjust="local", detach_opt:bool=False, num_samples:int=5, advx=None, dim_weight=None):
    assert relu_adjust in ["local"], "Only local relu adjustment is supported"
    optim_W, box_W = compute_tightness(abs_net, x, y, eps=eps, only_W=True, detach_opt=detach_opt, num_classes=num_classes, relu_adjust=relu_adjust, num_samples=num_samples) # W.shape = (bs, out_dim-1, in_dim), out_dim-1 due to last layer ellision
    if dim_weight is None:
        dim_weight = torch.ones(box_W.shape[:-1], device=box_W.device)
    reg = ((box_W - optim_W).sum(dim=-1) * dim_weight).sum(dim=-1).mean()
    return reg

def compute_vol_reg(abs_net, x, eps, bound_tol:float=0, recompute_box:bool=False, min_reg_eps=0, max_reg_eps=0.4, start_from:int=0):
    '''L = the area of relaxation triangles'''
    reg = 0
    reg_eps = max(min_reg_eps, min(eps, max_reg_eps))
    if recompute_box:
        abs_net.reset_bounds()
        # x = torch.clamp(x + 2 * (torch.rand_like(x, device=x.device) - 0.5) * (eps - reg_eps), min=0, max=1)
        x_abs = HybridZonotope.construct_from_noise(x, reg_eps, "box")
        abs_out = abs_net(x_abs)
        abs_out.concretize()
    for i, module in enumerate(abs_net.modules()):
        if i < start_from:
            continue
        if isinstance(module, ReLU):
            lower, upper = module.bounds
            bs = len(lower)
            # cross_mask = (lower <= 0) & (upper > 0)
            # reg += ((-lower)[cross_mask] * upper[cross_mask]).sum() / lower.numel()
            reg += ((-lower - bound_tol).clamp(min=0) * (upper - bound_tol).clamp(min=0)).view(bs, -1).sum(dim=1).mean()
            # unstable_lb_tol_exceed, unstable_ub_tol_exceed = ((-lower - bound_tol > 0) & (upper > 0)).float().mean().item(), ((upper - bound_tol > 0) & (lower < 0)).float().mean().item()
            # inactive_neuron, active_neuron = (upper < 0).float().mean().item(), (lower > 0).float().mean().item()
    return reg


def compute_fast_reg(abs_net, eps, tol=0.5):
    '''
    Ref: https://github.com/shizhouxing/Fast-Certified-Robust-Training/blob/addac383f6fac58d1bae8a231cf0ac9dab405a06/regularization.py

    loss = loss_tightness + loss_relu
    '''
    loss_tightness, loss_relu = 0, 0
    input_lower, input_upper = abs_net[1].bounds # net[0] is the normalization layer
    input_tightness = ((input_upper - input_lower) / 2).mean()
    cnt = 0
    for module in abs_net.modules():
        if isinstance(module, ReLU):
            # L_tightness
            lower, upper = module.bounds
            center = (upper + lower) / 2
            diff = ((upper - lower) / 2)
            tightness = diff.mean()
            mean_ = center.mean()

            loss_tightness += F.relu(tol - input_tightness / tightness.clamp(min=1e-12)) / tol

            mask_act, mask_inact = lower>0, upper<0
            mean_act = (center * mask_act).mean()
            mean_inact = (center * mask_inact).mean()
            delta = (center - mean_)**2
            var_act = (delta * mask_act).sum()
            var_inact = (delta * mask_inact).sum()

            mean_ratio = mean_act / -mean_inact
            var_ratio = var_act / var_inact
            mean_ratio = torch.min(mean_ratio, 1 / mean_ratio.clamp(min=1e-12))
            var_ratio = torch.min(var_ratio, 1 / var_ratio.clamp(min=1e-12))
            loss_relu_ = (F.relu(tol - mean_ratio) + F.relu(tol - var_ratio)) / tol
            if not torch.isnan(loss_relu_) and not torch.isinf(loss_relu_):
                loss_relu += loss_relu_ 
            cnt += 1
            
    loss = (loss_tightness + loss_relu) / cnt
    return loss