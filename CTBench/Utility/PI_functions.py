import torch
import torch.nn.functional as F
import time
import sys
import numpy as np
sys.path.append("../")
from AIDomains.abstract_layers import Sequential, Flatten, Linear, ReLU, Conv2d, _BatchNorm, BatchNorm2d, BatchNorm1d, Normalization
from AIDomains.zonotope import HybridZonotope
from AIDomains.ai_util import construct_C
from AIDomains.wrapper import propagate_abs

from typing import Optional
# torch.set_default_dtype(torch.float64)
from networks import get_network, fuse_BN_wrt_Flatten
from loaders import get_loaders
import argparse
from utils import seed_everything, round_sig

# def propagate_eps(input, net, C, abs:bool, relu_adjust=None):
#     for i, layer in enumerate(net):
#         if isinstance(layer, Normalization):
#             input = input / layer.sigma # the weight matrix is all positive
#         elif isinstance(layer, _BatchNorm):
#             w = (layer.weight / torch.sqrt(layer.current_var + layer.eps)).view(layer.view_dim)
#             if abs:
#                 input = input * w.abs()
#             else:
#                 input = input * w
#         elif isinstance(layer, Linear):
#             if i != len(net) - 1:
#                 if abs:
#                     input = F.linear(input, layer.weight.abs())
#                 else:
#                     input = F.linear(input, layer.weight)
#             else:
#                 # last linear, apply elision
#                 if abs:
#                     elided_weight = torch.matmul(C, layer.weight).abs()
#                 else:
#                     elided_weight = torch.matmul(C, layer.weight)
#                 input = torch.matmul(elided_weight, input.unsqueeze(-1)).squeeze(-1)
#         elif isinstance(layer, Conv2d):
#             if abs:
#                 input = F.conv2d(input, layer.weight.abs(), stride=layer.stride, padding=layer.padding, dilation=layer.dilation)
#             else:
#                 input = F.conv2d(input, layer.weight, stride=layer.stride, padding=layer.padding, dilation=layer.dilation)
#         elif isinstance(layer, Flatten):
#             input = input.flatten(1, -1)
#         elif isinstance(layer, ReLU):
#             if relu_adjust is None:
#                 pass
#             elif relu_adjust in ["local", "center", "random"]:
#                 lb, ub = layer.bounds
#                 deactivation = ub < 0
#                 input[deactivation] = 0
#             elif relu_adjust == "shrink":
#                 lb, ub = layer.bounds
#                 pre_size = ub - lb
#                 not_dead = ub > 0
#                 lb, ub = lb.clamp(min=0), ub.clamp(min=0)
#                 post_size = ub - lb
#                 input = input * (post_size.clamp(min=1e-8) / pre_size.clamp(min=1e-8)) * not_dead
#             else:
#                 raise NotImplementedError(f"Unknown ReLU adjustment: {relu_adjust}")
#         else:
#             raise NotImplementedError(f"Unknown layer: {layer}")
#     return input



# def compute_tightness2(net, batch_x, batch_y, eps, data_range=(0,1), num_classes:int=10, relu_adjust=None, detach_denom:bool=False, detach_num:bool=False, error_check:bool=False, verbose:bool=False):
#     '''
#     Warning: this would destroy the previous grad and stored box bounds for the net
#     '''
#     input_eps = ((batch_x+eps).clamp(max=data_range[1]) - (batch_x-eps).clamp(min=data_range[0])) / 2
#     num = input_eps.clone().detach()

#     if batch_y is None:
#         C = torch.eye(num_classes, device=batch_x.device).repeat(batch_x.shape[0], 1, 1)
#     else:
#         C = construct_C(num_classes, batch_y)

#     # BN_layers = [layer for layer in net if isinstance(layer, _BatchNorm)]
#     # original_stat = [layer.update_stat for layer in BN_layers]
#     # for layer in BN_layers:
#     #     layer.update_stat = False

#     # set relu adjustment here
#     # test status: correct. relu stat does not change inside this function after setting it below.
#     net.reset_bounds()
#     if relu_adjust == "local":
#         # use the activation pattern at the original input as the adjustment
#         with torch.no_grad():
#             abs_input = HybridZonotope.construct_from_noise(batch_x, 0, domain="box")
#             _ = net(abs_input)
#     elif relu_adjust == "center":
#         with torch.no_grad():
#             center = ((batch_x+eps).clamp(max=data_range[1]) + (batch_x-eps).clamp(min=data_range[0])) / 2
#             abs_input = HybridZonotope.construct_from_noise(center, 0, domain="box")
#             _ = net(abs_input)
#     elif relu_adjust == "random":
#         # use a random input as the adjustment
#         with torch.no_grad():
#             input_lb, input_ub = (batch_x-eps).clamp(min=data_range[0]), (batch_x+eps).clamp(max=data_range[1])
#             random_x = input_lb + (input_ub - input_lb) * torch.rand_like(batch_x, device=batch_x.device)
#             abs_input = HybridZonotope.construct_from_noise(random_x, 0, domain="box")
#             _ = net(abs_input)
#     elif relu_adjust == "shrink":
#         # for unstable neurons, shrink the coefficient to ensure the same box size
#         abs_input = HybridZonotope.construct_from_noise(batch_x, eps, domain="box")
#         _ = net(abs_input)

#     # infer signs of numerator here
#     with torch.enable_grad():
#         num.requires_grad = True
#         out = propagate_eps(num, net, C, abs=False, relu_adjust=relu_adjust)
#         net.zero_grad()
#         signs = []
#         out_dim = out.shape[-1]
#         for i in range(out_dim):
#             num.grad = None
#             # sum over batch because we only want the grad w.r.t. the batch eps which are unconnected
#             # thus, the grad of the sum is their individual grad
#             # test status: correct; tested via comparing the individual backward with it
#             out[..., i].sum().backward(retain_graph=True) 
#             signs.append(num.grad.sign())

#     # compute the numerator
#     # test status: no error found; tested via checking whether all num are the largest and positive
#     num = []
#     for i, sign in enumerate(signs):
#         num_one_dim = propagate_eps(input_eps * sign, net, C, abs=False, relu_adjust=relu_adjust)
#         num.append(num_one_dim)
#     num = torch.diagonal(torch.stack(num, dim=-1), dim1=-2, dim2=-1)

#     # compute the denominator
#     # test status: correct; tested via comparing direct propagation on a Deep Linear Network
#     # Numerical Problem with BN: result has <0.001% inconsistency
#     denom = propagate_eps(input_eps, net, C, abs=True, relu_adjust=relu_adjust)

#     if detach_num:
#         num = num.detach()
#     if detach_denom:
#         denom = denom.detach()

#     # print("num:", num)
#     # print("denom:", denom)

#     # abs_input = HybridZonotope.construct_from_noise(batch_x, eps, domain="box")
#     # abs_out = net(abs_input, C=C)
#     # lb, ub = abs_out.concretize()
#     # print("real:", (ub-lb)/2)


#     # for layer, stat in zip(BN_layers, original_stat):
#     #     layer.update_stat = stat

#     net.reset_bounds()
    
    # ratio = num.clamp(min=1e-8) / denom.clamp(min=1e-8)

    # if error_check and not (ratio <= 1.01).all():
    #     # numerical errors could lead to this;
    #     # enable error_check=True if this is strict
    #     mask = ratio > 1
    #     print(num[mask])
    #     print(denom[mask])
    #     torch.save(net, "buggie.ckpt")
    #     raise RuntimeError("PI > 1 detected.")
    # if verbose:
    #     return ratio, num, denom
    # else:
    #     return ratio

# def forward_weight_calc(y, net, input_dim, relu_adjust=None, num_classes:int=10):
#     '''
#     Does not support batch calculation due to memory limit of GPU.

#     This function assumes calculation on CPU.
#     '''
#     input_elements = np.prod(input_dim)
#     opt_W = torch.eye(input_elements).view(input_elements, *input_dim)
#     abs_W = opt_W.clone().detach()
#     tightness_list = []
#     current_box = HybridZonotope.construct_from_bounds(torch.zeros(1, *input_dim), 0.1+torch.zeros(1, *input_dim), domain="box")
#     for layer_id, layer in enumerate(net):
#         current_box = layer(current_box)
#         if isinstance(layer, Normalization):
#             opt_W = opt_W / layer.sigma
#             abs_W = abs_W / layer.sigma
#         elif isinstance(layer, _BatchNorm):
#             w = (layer.weight / torch.sqrt(layer.current_var + layer.eps)).view(layer.view_dim)
#             opt_W = opt_W * w
#             abs_W = abs_W * w.abs()
#         elif isinstance(layer, Flatten):
#             opt_W = opt_W.flatten(start_dim=1)
#             abs_W = abs_W.flatten(start_dim=1)
#         elif isinstance(layer, Linear):
#             if not layer_id == len(net) - 1:
#                 opt_W = torch.matmul(opt_W, layer.weight.T)
#                 abs_W = torch.matmul(abs_W, layer.weight.abs().T)
#             else:
#                 # last linear, apply elision
#                 C = construct_C(num_classes, y).squeeze(0)
#                 weight = torch.matmul(C, layer.weight)
#                 opt_W = torch.matmul(opt_W, weight.T)
#                 abs_W = torch.matmul(abs_W, weight.abs().T)
#         elif isinstance(layer, Conv2d):
#             # TODO: wrong result
#             # TODO: abs_W: first conv2d is correct, but the rest are wrong
#             print(opt_W.shape)
#             opt_W = F.conv2d(opt_W, layer.weight, stride=layer.stride, padding=layer.padding, dilation=layer.dilation, groups=layer.groups)
#             abs_W = F.conv2d(abs_W, layer.weight.abs(), stride=layer.stride, padding=layer.padding, dilation=layer.dilation, groups=layer.groups)
#             # TODO: delete check
#             lb, ub = current_box.concretize()
#             print(ub.shape, abs_W.shape)
#             print(((ub - lb) / 2 - (0.05*abs_W.sum(dim=0, keepdims=True))).mean())
#             # raise
#         elif isinstance(layer, ReLU):
#             pass
#         else:
#             raise NotImplementedError(f"Unknown layer: {layer}")
#         if layer_id == len(net)-1 or isinstance(net[layer_id+1], ReLU):
#             tightness_list.append((opt_W.abs().clamp(min=1e-8) / abs_W.clamp(min=1e-8)).mean().item())
#     return tightness_list

def backward_weight_calc(C, net, abs:bool, relu_adjust=None):
    cur_W = C.clone().detach()
    for rev_idx, layer in enumerate(net[::-1]):
        if isinstance(layer, Linear):
            if rev_idx == 0:
                cur_W = torch.matmul(cur_W, layer.weight)
                if abs:
                    cur_W = cur_W.abs()
            else:
                if abs:
                    cur_W = torch.matmul(cur_W, layer.weight.abs())
                else:
                    cur_W = torch.matmul(cur_W, layer.weight)
        elif isinstance(layer, ReLU):
            if relu_adjust in ["local", "center", "dead", "random_value_avg"]:
                lb, ub = layer.bounds
                activated = (lb >= 0).unsqueeze(1)
                cur_W = cur_W * activated.float()
        elif isinstance(layer, Flatten):
            in_dim = net[len(net)-rev_idx-2].output_dim
            cur_W = cur_W.view(cur_W.shape[0], -1, *in_dim)
        elif isinstance(layer, Conv2d):
            # merge the batch dim and output dim
            bs = cur_W.shape[0]
            cur_W = cur_W.view(-1, *cur_W.shape[2:])
            in_dim = net[len(net)-rev_idx-2].output_dim
            w_padding = (
                in_dim[1]
                + 2 * layer.padding[0]
                - 1
                - layer.dilation[0] * (layer.weight.shape[-2] - 1)
            ) % layer.stride[0]
            h_padding = (
                in_dim[2]
                + 2 * layer.padding[1]
                - 1
                - layer.dilation[1] * (layer.weight.shape[-1] - 1)
            ) % layer.stride[1]
            output_padding = (w_padding, h_padding)

            weight = layer.weight if not abs else layer.weight.abs()
            cur_W = F.conv_transpose2d(cur_W, weight, stride=layer.stride, padding=layer.padding, output_padding=output_padding, groups=layer.groups, dilation=layer.dilation)

            # unmerge the batch dim and output dim: leads to wasted view operation but should be OK
            cur_W = cur_W.view(bs, -1, *cur_W.shape[1:])
        elif isinstance(layer, Normalization):
            # merge the batch dim and output dim
            bs = cur_W.shape[0]
            cur_W = cur_W.view(-1, *cur_W.shape[2:])
            cur_W = cur_W / layer.sigma
            # unmerge the batch dim and output dim: leads to wasted view operation but should be OK
            cur_W = cur_W.view(bs, -1, *cur_W.shape[1:])
        elif isinstance(layer, _BatchNorm):
            # merge the batch dim and output dim
            bs = cur_W.shape[0]
            cur_W = cur_W.view(-1, *cur_W.shape[2:])
            w = (layer.weight / torch.sqrt(layer.current_var + layer.eps)).view(layer.view_dim)
            if abs:
                w = w.abs()
            cur_W = cur_W * w
            # unmerge the batch dim and output dim: leads to wasted view operation but should be OK
            cur_W = cur_W.view(bs, -1, *cur_W.shape[1:])
    cur_W = cur_W.flatten(start_dim=2).abs()
    return cur_W

def compute_all_layer_tightness(net, batch_x, batch_y, eps:float=None, only_W:bool=False, detach_opt:bool=False, data_range=(0,1), num_classes:int=10, relu_adjust=None, num_samples:int=5, error_check:bool=False, verbose:bool=False):
    '''
    Compute tightness for all subnets before each ReLU layer. Gradients are not preserved.
    '''
    assert len(batch_x) == 1, "batch size must be 1 for compute_all_layer_tightness"
    relu_pos = [i for i, layer in enumerate(net) if isinstance(layer, ReLU)]
    net_split = [Sequential(*net[:pos], Flatten()) for pos in relu_pos]
    tightness_list = []
    with torch.no_grad():
        for subnet in net_split:
            subnet.set_dim(torch.zeros((len(batch_x), *input_dim), device=device))
            tightness = compute_tightness(subnet, batch_x, None, eps, only_W, detach_opt, data_range, np.prod(subnet[-2].output_dim), relu_adjust, num_samples, error_check, verbose=False)
            tightness_list.append(tightness.mean().item())
        # compute end to end tightness
        tightness = compute_tightness(net, batch_x, batch_y, eps, only_W, detach_opt, data_range, num_classes, relu_adjust, num_samples, error_check, verbose=False)
        tightness_list.append(tightness.mean().item())
    return torch.tensor(tightness_list, device=device)

def compute_tightness(net, batch_x, batch_y, eps:float=None, only_W:bool=False, data_range=(0,1), num_classes:int=10, relu_adjust=None, num_samples:int=5, error_check:bool=False, verbose:bool=False, max_examined_class:Optional[int]=None):
    '''
    Compute end to end propagation tightness of the given network considering elision of the last layer, i.e., Interval Bound Propagation.

    Warning: this would destroy the stored box bounds for the net

    @remark
        When num_classes-1 (-1 due to margin computation) exceeds max_examined_class, only max_examined_class classes (selected randomly) will be examined. The selected classes vary across the batch dimension for unbiasedness. When this feature is invoked, the result will be a statistically unbiased estimate of the PI, but the variance will be higher (increased by `max_examined_class/(num_class-1)` times). However, the variance is usually super small, especially as propagation tightness is computed w.r.t. the full dataset, so this feature is recommended for large num_classes.
    '''
    if batch_y is None:
        C = torch.eye(num_classes, device=batch_x.device).repeat(batch_x.shape[0], 1, 1)
    else:
        C = construct_C(num_classes, batch_y)

    if max_examined_class is not None:
        # C is a small matrix, so we can copy it with a for loop; for C.shape=(batch_size, num_class-1, num_class)=(128, 199, 200), it took 0.02 seconds for the loop below
        # We do this because torch does not have a known batched multinomial function
        selected_idx = [torch.multinomial(torch.ones(C.shape[1], device=C.device), max_examined_class, replacement=False) for _ in range(C.shape[0])]
        C = torch.stack([C[i, idx] for i, idx in enumerate(selected_idx)], dim=0)

    # set relu adjustment here
    # test status: correct. relu stat does not change inside this function after setting it below.
    net.reset_bounds()
    if relu_adjust == "local":
        # use the activation pattern at the original input as the adjustment
        with torch.no_grad():
            abs_input = HybridZonotope.construct_from_noise(batch_x, 0, domain="box")
            _ = net(abs_input)
    elif relu_adjust == "center":
        # use the activation pattern at the center of the input as the adjustment
        with torch.no_grad():
            center = ((batch_x+eps).clamp(max=data_range[1]) + (batch_x-eps).clamp(min=data_range[0])) / 2
            abs_input = HybridZonotope.construct_from_noise(center, 0, domain="box")
            _ = net(abs_input)
    elif relu_adjust == "dead":
        # only deactivate the dead neurons
        with torch.no_grad():
            abs_input = HybridZonotope.construct_from_noise(batch_x, eps, domain="box")
            _ = net(abs_input)
    elif relu_adjust is None:
        pass
    else:
        raise NotImplementedError(f"Unknown ReLU adjustment: {relu_adjust}")
    
    # Compute num_W = |\Prod_i W_i| and denom_W = \Prod_i |W_i|
    num_W = backward_weight_calc(C, net, abs=False, relu_adjust=relu_adjust)
    denom_W = backward_weight_calc(C, net, abs=True, relu_adjust=relu_adjust)

    if only_W:
        return num_W, denom_W
    else:
        assert eps is not None, "eps must be provided if only_W=False"

    input_eps = ((batch_x+eps).clamp(max=data_range[1]) - (batch_x-eps).clamp(min=data_range[0])) / 2
    input_eps = input_eps.flatten(start_dim=1).unsqueeze(-1)
    num = torch.matmul(num_W, input_eps).squeeze(-1)
    denom = torch.matmul(denom_W, input_eps).squeeze(-1)
    ratio = num.clamp(min=1e-8) / denom.clamp(min=1e-8)

    if error_check and not (ratio <= 1.01).all():
        # numerical errors could lead to this;
        # enable error_check=True if this is strict
        mask = ratio > 1
        print(num[mask])
        print(denom[mask])
        torch.save(net, "buggie.ckpt")
        raise RuntimeError("PI > 1 detected.")
    
    if verbose:
        return ratio, num, denom
    else:
        return ratio

if __name__ == "__main__":
    seed_everything(1)
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.dataset = "cifar10"
    args.train_batch = 1
    args.test_batch = 1
    args.grad_accu_batch = None
    args.frac_valid = None
    args.net = "cnn_5layer"
    args.init = "default"

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = "cpu"
    loaders, input_size, input_channel, n_class = get_loaders(args, shuffle_test=False)
    train_loader, test_loader = loaders
    input_dim = (input_channel, input_size, input_size)

    net = get_network(args.net, args.dataset, device, init=args.init)
    net = Sequential.from_concrete_network(net, input_dim, disconnect=True)
    net.load_state_dict(torch.load("../../PI-master/test_models/seed_123/cifar10/eps0.0078431/box_trained/cnn_5layer/init_fast/alpha5.0/certW_1.0/anneal_80/fast_reg/partial_PI_0_target_0.5/model.ckpt"))
    # net.load_state_dict(torch.load("../../PI-master/test_models/seed_123/cifar10/eps0.0078431/box_trained/cnn_5layer_bn/init_fast/alpha5.0/certW_1.0/anneal_80/fast_reg/partial_PI_0_target_0.5/model.ckpt"))

    # net = Sequential(Conv2d(1, 4, (3,3), padding=(1,1), stride=(2,2)))

    net.set_dim(torch.zeros((test_loader.batch_size, *input_dim), device=device))

    print(net)

    all_tightness = []
    for i, (x, y) in enumerate(test_loader):
        x = x.to(device)
        y = y.to(device)
        # print(forward_weight_calc(y, net, input_dim, relu_adjust=None))
        tightness = compute_all_layer_tightness(net, x, y, eps=0.1, data_range=(0,1), num_classes=n_class, relu_adjust="local", verbose=False)
        all_tightness.append(tightness)
        # print(compute_tightness(net, x, y, eps=0.1, data_range=(0,1), num_classes=n_class, relu_adjust=None, verbose=False).mean().item())
        if i == 500:
            break
    all_tightness = torch.stack(all_tightness, dim=0)
    print(all_tightness.mean(dim=0))
    print(all_tightness.std(dim=0))

    # eps = 0.2
    # for x, y in test_loader:
    #     x = x.to(device)
    #     y = y.to(device)
    #     t1 = time.time()
    #     ratio2, num2, denom2 = compute_tightness(net, x, y, eps, relu_adjust="random_value_avg", verbose=True)
    #     t2 = time.time()
    #     print(f"Time: {t2-t1:.2E}s")
    #     print(torch.cuda.max_memory_allocated() / (1024**3))
    #     print(ratio2)
    #     print(num2)
    #     print(denom2)
    #     raise
