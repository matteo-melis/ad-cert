'''
Unit test for model_wrapper.py
'''
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import time
import unittest
import sys
sys.path.append("..")
from model_wrapper import get_model_wrapper, BasicModelWrapper, PGDModelWrapper, MultiPGDModelWrapper, BoxModelWrapper, TAPSModelWrapper, SmallBoxModelWrapper, STAPSModelWrapper, DeepPolyModelWrapper, ARoWModelWrapper, MARTModelWrapper, MTLIBPModelWrapper, CCIBPModelWrapper, EXPIBPModelWrapper, BasicFunctionWrapper, GradAccuFunctionWrapper, WeightSmoothFunctionWrapper, SAMFunctionWrapper
from AIDomains.abstract_layers import Sequential, _BatchNorm, Linear, Conv2d
import AIDomains.abstract_layers as abs_layers
from utils import seed_everything
from bunch import Bunch

class Argument:
    def __init__(self, **kwargs) -> None:
        args = Bunch()
        args = self.default_args(args)
        for k, v in kwargs.items():
            setattr(args, k, v)
        self.args = args

    def parse(self):
        return self.args
    
    def default_args(self, args):
        args.step_size = None
        args.restarts = 2
        args.train_steps = 10
        args.test_eps = 0.1
        args.estimation_batch = 4
        args.soft_thre = 0.5
        args.taps_grad_scale = 1
        args.no_ibp_multiplier = False
        args.eps_shrinkage = 0.6
        args.relu_shrinkage = 0.2
        args.disable_TAPS = False
        args.model_selection = None
        return args

def is_None_or_zero(x):
    if isinstance(x, torch.Tensor):
        return (x == 0).all()
    return x is None or x == 0

def has_no_grad(model):
    flag = True
    for p in model.parameters():
        if not is_None_or_zero(p.grad):
            flag = False
            break
    return flag

def get_test_network(include_BN:bool=True, input_dim=(2,2,2), num_class=10, small:bool=False):
    if not small:
        layers = [nn.Conv2d(input_dim[0], 16, 3, 1, 1)]
        if include_BN:
            layers.append(nn.BatchNorm2d(16))
        layers.append(nn.Flatten())
        layers.append(nn.Linear(16*np.prod(input_dim[1:]), 2*num_class))
        layers.append(nn.ReLU())
        if include_BN:
            layers.append(nn.BatchNorm1d(2*num_class))
        layers.append(nn.Linear(2*num_class, num_class))
    else:
        layers = [nn.Flatten()]
        layers.append(nn.Linear(np.prod(input_dim), num_class))
        if include_BN:
            layers.append(nn.BatchNorm1d(num_class))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(num_class, num_class))
    net = nn.Sequential(*layers)
    net = Sequential.from_concrete_network(net, input_dim, disconnect=True)
    net.set_dim(torch.zeros(1, *input_dim))
    return net, input_dim, num_class

class Tester(unittest.TestCase):
    def _test_set_value_type(self, obj, attr:str, disallow_type:tuple, test_values:tuple):
        '''
        Try to set the attribute of obj to different types of values and check whether the setting is as expected.

        If test_values is not None, then successful setting will be tested on those values.
        '''
        assert hasattr(obj, attr), f"{obj} does not have attribute {attr}"
        value_dict = {
            "numeric": (-1.0, 2, 3.0),
            "NoneType": (None, ),
            "str": ("a", "bcd"),
            "bool": (False, True)
        }
        for value in test_values:
            setattr(obj, attr, value)
            self.assertEqual(getattr(obj, attr), value)
        for value_type in disallow_type:
            for value in value_dict[value_type]:
                with self.assertRaises(Exception):
                    setattr(obj, attr, value)

    def test_property(self):
        net, input_dim, num_class = get_test_network()
        wrapper = self._get_model_wrapper_to_test()
        # robust weight
        self.assertEqual(wrapper.robust_weight, None) # initial value
        self._test_set_value_type(wrapper, "robust_weight", ["str", "NoneType"], (0., 0.5, 1.))
        with self.assertRaises(AssertionError):
            wrapper.robust_weight = 1.1 # robust weight should be in [0, 1]
        with self.assertRaises(AssertionError):
            wrapper.robust_weight = -0.1
        # summary accu stat
        self.assertEqual(wrapper.summary_accu_stat, True) # initial value
        self._test_set_value_type(wrapper, "summary_accu_stat", ("str", "numeric", "NoneType"), (False, True))
        # data min / max
        self.assertEqual(wrapper.data_min, 0); self.assertEqual(wrapper.data_max, 1) # initial value
        self.assertGreaterEqual(wrapper.data_max, wrapper.data_min)
        with self.assertRaises(AssertionError):
            wrapper.data_max = wrapper.data_min - 1
        with self.assertRaises(AssertionError):
            wrapper.data_min = wrapper.data_max + 1
        self._test_set_value_type(wrapper, "data_min", ["str", "NoneType"], (-1.0, 0.0))
        self._test_set_value_type(wrapper, "data_max", ["str", "NoneType"], (2.0, 3.0))
        # freeze BN
        self._test_set_value_type(wrapper, "freeze_BN", ("str", "numeric", "NoneType"), (False, True))
        # current_eps and max_eps
        self.assertEqual(wrapper.current_eps, 0); self.assertEqual(wrapper.max_eps, 0) # initial value
        with self.assertRaises(AssertionError):
            wrapper.current_eps = 0.1 # since max_eps is 0
        self._test_set_value_type(wrapper, "max_eps",["str", "NoneType"], (0.2, 0.3))
        self._test_set_value_type(wrapper, "current_eps",["str", "NoneType"], (0.2, 0.3))
        with self.assertRaises(AssertionError):
            wrapper.max_eps = 0.01 # max eps cannot be set smaller than current eps
        # current lr
        self.assertEqual(wrapper.current_lr, None) # initial value
        self._test_set_value_type(wrapper, "current_lr", ["str", "NoneType"], (0.1, 0.2))
        # grad_cleaner
        self._test_set_value_type(wrapper, "grad_cleaner", "", ()) # read-only
        # device
        self._test_set_value_type(wrapper, "device", ["str", "numeric", "bool", "NoneType"], ())
        wrapper = wrapper.to("cuda")
        self.assertEqual(wrapper.device, "cuda")
        for p in wrapper.net.parameters():
            self.assertEqual(p.device, torch.device("cuda:0"))
        wrapper = wrapper.to("cpu")
        self.assertEqual(wrapper.device, "cpu")
        for p in wrapper.net.parameters():
            self.assertEqual(p.device, torch.device("cpu"))

    def _get_model_wrapper_to_test(self):
        device = "cpu"
        net, input_dim, num_class = get_test_network()
        wrapper = BasicModelWrapper(net, nn.CrossEntropyLoss(), input_dim, device, None)
        return wrapper

    def _get_random_input(self, input_dim=(2,2,2), device="cpu", num_class=10, bs=16):
        x = torch.rand(bs, *input_dim).to(device)
        y = torch.randint(num_class, (bs, )).flatten().to(device)
        return x, y

    def test_compute_nat_loss_and_set_BN(self):
        device = "cpu"
        net, input_dim, num_class = get_test_network()
        wrapper = self._get_model_wrapper_to_test()
        wrapper.net.train()
        x, y = self._get_random_input(input_dim, device, num_class)
        # nat loss should have no grad_fn when robust_weight = 1
        wrapper.robust_weight = 1
        loss, accu, pred_correct = wrapper.compute_nat_loss_and_set_BN(x, y)
        self.assertTrue(has_no_grad(wrapper.net))
        with self.assertRaises(Exception):
            loss.backward()
        wrapper.robust_weight = 0
        loss, accu, pred_correct = wrapper.compute_nat_loss_and_set_BN(x, y)
        self.assertTrue(has_no_grad(wrapper.net))
        loss.backward()
        self.assertFalse(has_no_grad(wrapper.net))
        # test grad cleaner
        wrapper.grad_cleaner.zero_grad()
        self.assertTrue(has_no_grad(wrapper.net))
        # test batch norm
        BN_layers = [layer for layer in wrapper.net if isinstance(layer, _BatchNorm)]
        self.assertTrue(BN_layers[-1].current_mean.any())
        self.assertTrue(BN_layers[-1].current_var.any())
        # after eval mode, the current mean and var should be the same as running mean and var
        wrapper.net.eval()
        loss, accu, pred_correct = wrapper.compute_nat_loss_and_set_BN(x, y)
        self.assertTrue((BN_layers[-1].current_mean - BN_layers[-1].running_mean).abs().max() < 1e-6)
        self.assertTrue((BN_layers[-1].current_var - BN_layers[-1].running_var).abs().max() < 1e-6)
        # test freeze BN
        old_mean = BN_layers[-1].current_mean.clone().detach()
        wrapper.net.train()
        wrapper.freeze_BN = True
        loss, accu, pred_correct = wrapper.compute_nat_loss_and_set_BN(x, y)
        self.assertTrue((BN_layers[-1].current_mean == old_mean).all())
        wrapper.freeze_BN = False
        loss, accu, pred_correct = wrapper.compute_nat_loss_and_set_BN(x, y)
        self.assertTrue((BN_layers[-1].current_mean != old_mean).any())

    def test_convert_pred_to_margin(self):
        bs = 16
        num_class = 10
        pred = torch.randn(bs, num_class)
        x, y = self._get_random_input(bs=bs)
        wrapper = self._get_model_wrapper_to_test()
        # check functionality when rearrange label = False
        margin = wrapper.convert_pred_to_margin(y, pred, rearrange_label=False)
        for i in range(bs):
            for j in range(num_class):
                self.assertEqual(margin[i, j], pred[i, j] - pred[i, y[i]])
        # check functionality when rearrange label = True
        margin = wrapper.convert_pred_to_margin(y, pred, rearrange_label=True)
        for i in range(bs):
            for j in range(num_class):
                if j == 0:
                    self.assertEqual(margin[i, j], 0)
                else:
                    if j == y[i]:
                        continue
                    nj = j - 1 if j < y[i] else j
                    self.assertEqual(margin[i, j], pred[i, nj] - pred[i, y[i]])
        self.assertTrue(has_no_grad(wrapper.net))

class TestBasicModelWrapper(Tester):
    def test_compute_model_stat(self):
        x, y = self._get_random_input()
        wrapper = self._get_model_wrapper_to_test()
        wrapper.summary_accu_stat = True
        # normal behavior
        eps = 0
        (loss, nat_loss, robust_loss), (nat_accu, robust_accu) = wrapper.compute_model_stat(x, y, eps)
        self.assertEqual(nat_loss, loss)
        self.assertEqual(robust_loss, 0)
        self.assertEqual(nat_accu, robust_accu)
        self.assertTrue(has_no_grad(wrapper.net))
        # reject eps > 0 for BasicModelWrapper
        eps = 0.1
        with self.assertRaises(Exception):
            wrapper.compute_model_stat(x, y, eps)


class TestPGDModelWrapper(Tester):
    def _get_model_wrapper_to_test(self):
        device = "cpu"
        net, input_dim, num_class = get_test_network(small=True)
        args = Argument(test_eps=0).parse()
        wrapper = PGDModelWrapper(net, nn.CrossEntropyLoss(), input_dim, device, args, enable_EDAC=False)
        return wrapper

    def test_property(self):
        super().test_property()
        wrapper = self._get_model_wrapper_to_test()
        wrapper.register_EDAC_hyperparam(None, 0.1)
        # EDAC step size
        self._test_set_value_type(wrapper, "EDAC_step_size", ("str", "NoneType"), (0.1, 0.2))
        # EDAC optimizer
        self._test_set_value_type(wrapper, "EDAC_optimizer", ("str", "numeric", "NoneType"), ())
        # num steps
        self._test_set_value_type(wrapper, "num_steps", ("str", "NoneType"), (1, 2))
        with self.assertRaises(AssertionError):
            wrapper.num_steps = 0
        # restarts
        self._test_set_value_type(wrapper, "restarts", ("str", "NoneType"), (1, 2))
        with self.assertRaises(AssertionError):
            wrapper.restarts = 0
        # enable EDAC
        self._test_set_value_type(wrapper, "enable_EDAC", ("str", "numeric", "NoneType"), (False, True))
        # cache adv xy
        self._test_set_value_type(wrapper, "cache_advx", ("str", "numeric", "NoneType"), (False, True))
        self._test_set_value_type(wrapper, "cache_advy", ("str", "numeric", "NoneType"), (False, True))

    def test_get_robust_stat_from_bounds(self):
        x, y = self._get_random_input()
        wrapper = self._get_model_wrapper_to_test()
        wrapper.robust_weight = 1
        wrapper.summary_accu_stat = True
        eps = 0.1
        wrapper.get_robust_stat_from_bounds((x-eps).clamp(0, 1), (x+eps).clamp(0, 1), x, y)
        self.assertTrue(has_no_grad(wrapper.net))

    def test_compute_model_stat(self, sanity_check:bool=True):
        x, y = self._get_random_input()
        wrapper = self._get_model_wrapper_to_test()
        wrapper.enable_EDAC = False
        wrapper.robust_weight = 1
        wrapper.summary_accu_stat = True
        eps = 0.1
        wrapper.max_eps = 0.1
        # test current eps
        self.assertEqual(wrapper.current_eps, 0.)
        self.assertTrue(has_no_grad(wrapper.net))
        (loss, nat_loss, robust_loss), (nat_accu, robust_accu) = wrapper.compute_model_stat(x, y, eps)
        self.assertEqual(wrapper.current_eps, eps)
        # test whether compute_model_stat can change grad
        self.assertTrue(has_no_grad(wrapper.net))
        self.assertFalse(nat_loss.requires_grad)
        # check loss backward will compute grad
        loss.backward()
        self.assertFalse(has_no_grad(wrapper.net))
        wrapper.grad_cleaner.zero_grad()
        # test combine loss
        for w in [0, 1, *(np.random.rand(10))]:
            wrapper.robust_weight = w
            (loss, nat_loss, robust_loss), (nat_accu, robust_accu) = wrapper.compute_model_stat(x, y, eps)
            self.assertEqual(loss, robust_loss*w+nat_loss*(1-w))
        # test cache_adv_xy_if_need
        self.assertFalse(wrapper.cache_advx)
        self.assertFalse(wrapper.cache_advy)
        self.assertIsNone(wrapper.current_advx)
        self.assertIsNone(wrapper.current_advy)
        wrapper.cache_advx = wrapper.cache_advy = True
        wrapper.net.train()
        (loss, nat_loss, robust_loss), (nat_accu, robust_accu) = wrapper.compute_model_stat(x, y, eps)
        self.assertIsNotNone(wrapper.current_advx)
        self.assertIsNotNone(wrapper.current_advy)
        self.assertTrue((wrapper.net(wrapper.current_advx) == wrapper.current_advy).all())
        wrapper.cache_advx = wrapper.cache_advy = False
        if sanity_check:
            # sanity check when eps = 0
            eps = 0
            wrapper.net.eval()
            (loss, nat_loss, robust_loss), (nat_accu, robust_accu) = wrapper.compute_model_stat(x, y, eps)
            self.assertIsNone(wrapper.current_advx)
            self.assertIsNone(wrapper.current_advy)
            self.assertAlmostEqual(nat_loss.item(), robust_loss.item(), delta=1e-6)
            self.assertEqual(nat_accu, robust_accu)
            self.assertTrue(has_no_grad(wrapper.net))
        # test whether DEAC works
        wrapper.enable_EDAC = True
        wrapper.num_steps = 1
        wrapper.restarts = 1
        wrapper.register_EDAC_hyperparam(torch.optim.SGD(wrapper.net.parameters(), lr=1e-2), 0.1)
        (loss, nat_loss, robust_loss), (nat_accu, robust_accu) = wrapper.compute_model_stat(x, y, eps)



class TestARoWModelWrapper(TestPGDModelWrapper):
    def _get_model_wrapper_to_test(self):
        args = Argument(train_steps=10, test_eps=0, restarts=2).parse()
        device = "cpu"
        net, input_dim, num_class = get_test_network(small=True)
        wrapper = ARoWModelWrapper(net, nn.CrossEntropyLoss(), input_dim, device, args, enable_EDAC=True)
        return wrapper
    
    def test_property(self):
        super().test_property()
        wrapper = self._get_model_wrapper_to_test()
        # reg weight
        self._test_set_value_type(wrapper, "reg_weight", ("str", "NoneType"), (0.1, 0.2))
        # label smoothing
        self._test_set_value_type(wrapper.LS_loss_fn, "smoothing", ("str", "NoneType"), (0.1, 0.2))
        with self.assertRaises(AssertionError):
            wrapper.LS_loss_fn.smoothing = -1

    def test_compute_model_stat(self):
        super().test_compute_model_stat(sanity_check=False) # additional loss is added to robust loss so eps=0 does not mean nat_loss=robust_loss

class TestMARTModelWrapper(TestPGDModelWrapper):
    def _get_model_wrapper_to_test(self):
        device = "cpu"
        net, input_dim, num_class = get_test_network(small=True)
        args = Argument(test_eps=0).parse()
        wrapper = MARTModelWrapper(net, nn.CrossEntropyLoss(), input_dim, device, args, enable_EDAC=False)
        return wrapper
    
    def test_property(self):
        super().test_property()
        wrapper = self._get_model_wrapper_to_test()
        # reg weight
        self._test_set_value_type(wrapper, "reg_weight", ("str", "NoneType"), (0.1, 0.2))
        with self.assertRaises(AssertionError):
            wrapper.reg_weight = -1

    def test_compute_model_stat(self):
        super().test_compute_model_stat(sanity_check=False) # additional loss is added to robust loss so eps=0 does not mean nat_loss=robust_loss

class TestBoxModelWrapper(Tester):
    def _get_model_wrapper_to_test(self):
        device = "cpu"
        net, input_dim, num_class = get_test_network(small=True)
        args = Argument(test_eps=0).parse()
        wrapper = BoxModelWrapper(net, nn.CrossEntropyLoss(), input_dim, device, args)
        return wrapper
    
    def test_property(self):
        super().test_property()
        wrapper = self._get_model_wrapper_to_test()
        # store box bounds
        self._test_set_value_type(wrapper, "store_box_bounds", ("str", "numeric", "NoneType"), (False, True))
        # relu_shrinkage
        self._test_set_value_type(wrapper, "relu_shrinkage", ("str", "numeric", "NoneType"), ())

    def test_compute_model_stat(self):
        x, y = self._get_random_input()
        wrapper = self._get_model_wrapper_to_test()
        wrapper.net.train()
        wrapper.max_eps = 0.2
        # normal test
        eps = 0.1
        wrapper.robust_weight = 1
        wrapper.store_box_bounds = False
        (loss, nat_loss, robust_loss), (nat_accu, robust_accu) = wrapper.compute_model_stat(x, y, eps)
        self.assertEqual(wrapper.current_eps, eps)
        self.assertTrue(has_no_grad(wrapper.net))
        loss.backward()
        self.assertFalse(has_no_grad(wrapper.net))
        wrapper.grad_cleaner.zero_grad()
        # function test for store_box_bounds
        for layer in wrapper.net:
            if isinstance(layer, abs_layers.ReLU):
                self.assertIsNone(layer.bounds)
        # consistency test for eps=0
        wrapper.store_box_bounds = True
        eps = 0
        (loss, nat_loss, robust_loss), (nat_accu, robust_accu) = wrapper.compute_model_stat(x, y, eps)
        self.assertAlmostEqual(nat_loss.item(), robust_loss.item(), delta=1e-6)
        self.assertEqual(nat_accu, robust_accu)
        for layer in wrapper.net:
            if isinstance(layer, abs_layers.ReLU):
                self.assertTrue((layer.bounds[0] == layer.bounds[1]).all())
        self.assertTrue(has_no_grad(wrapper.net))

    
class TestDeepPolyModelWrapper(Tester):
    def _get_model_wrapper_to_test(self):
        device = "cpu"
        net, input_dim, num_class = get_test_network(small=True)
        args = Argument(test_eps=0).parse()
        wrapper = DeepPolyModelWrapper(net, nn.CrossEntropyLoss(), input_dim, device, args)
        return wrapper
    
    def test_property(self):
        super().test_property()
        wrapper = self._get_model_wrapper_to_test()
        wrapper.max_eps = 0.2
        # use dp box
        self._test_set_value_type(wrapper, "use_dp_box", ("str", "numeric", "NoneType"), (False, True))
        # loss fusion
        self._test_set_value_type(wrapper, "loss_fusion", ("str", "numeric", "NoneType"), (False, True))

    def test_compute_model_stat(self):
        x, y = self._get_random_input()
        wrapper = self._get_model_wrapper_to_test()
        wrapper.max_eps = 0.2
        wrapper.net.train()
        # normal test
        eps = 0.1
        wrapper.robust_weight = 1
        (loss, nat_loss, robust_loss), (nat_accu, robust_accu) = wrapper.compute_model_stat(x, y, eps)
        self.assertEqual(wrapper.current_eps, eps)
        self.assertTrue(has_no_grad(wrapper.net))
        loss.backward()
        self.assertFalse(has_no_grad(wrapper.net))
        wrapper.grad_cleaner.zero_grad()
        # consistency test for eps=0 CROWN-IBP
        wrapper.use_dp_box = True
        eps = 0
        (loss, nat_loss, robust_loss), (nat_accu, robust_accu) = wrapper.compute_model_stat(x, y, eps)
        self.assertAlmostEqual(nat_loss.item(), robust_loss.item(), delta=1e-6)
        self.assertEqual(nat_accu, robust_accu)
        for layer in wrapper.net:
            if isinstance(layer, abs_layers.ReLU):
                self.assertLess((layer.bounds[0] - layer.bounds[1]).abs().max(), 1e-6)
        # function test for reset_bounds
        wrapper.net.reset_bounds()
        for layer in wrapper.net:
            if isinstance(layer, abs_layers.ReLU):
                self.assertIsNone(layer.bounds)
        # consistency test for eps=0 DeepPoly
        wrapper.use_dp_box = False
        eps = 0
        (loss, nat_loss, robust_loss), (nat_accu, robust_accu) = wrapper.compute_model_stat(x, y, eps)
        self.assertAlmostEqual(nat_loss.item(), robust_loss.item(), delta=1e-6)
        self.assertEqual(nat_accu, robust_accu)
        for layer in wrapper.net:
            if isinstance(layer, abs_layers.ReLU):
                self.assertLess((layer.bounds[0] - layer.bounds[1]).abs().max(), 1e-6)
        self.assertTrue(has_no_grad(wrapper.net))

    def test_nonnegative_weight(self):
        # for nonnegative net, DPBox should equivalent to DeepPoly
        x, y = self._get_random_input()
        wrapper = self._get_model_wrapper_to_test()
        wrapper.max_eps = 0.2
        wrapper.net.eval()
        for layer in wrapper.net:
            if hasattr(layer, "weight"):
                layer.weight.data = torch.abs(layer.weight.data)
        wrapper.robust_weight = 1
        with torch.no_grad():
            eps = 0.1
            wrapper.use_dp_box = True
            (loss1, nat_loss1, robust_loss1), (nat_accu1, robust_accu1) = wrapper.compute_model_stat(x, y, eps)
            wrapper.net.reset_bounds()
            wrapper.use_dp_box = False
            (loss2, nat_loss2, robust_loss2), (nat_accu2, robust_accu2) = wrapper.compute_model_stat(x, y, eps)
            self.assertAlmostEqual(loss1.item(), loss2.item(), delta=1e-5)
            self.assertEqual(nat_loss1.item(), nat_loss2.item())
            self.assertAlmostEqual(robust_loss1.item(), robust_loss2.item(), delta=1e-5)
            self.assertEqual(nat_accu1, nat_accu2)
            self.assertEqual(robust_accu1, robust_accu2)
        self.assertTrue(has_no_grad(wrapper.net))


class TestMultiPGDModelWrapper(TestPGDModelWrapper):
    def _get_model_wrapper_to_test(self):
        device = "cpu"
        net, input_dim, num_class = get_test_network(small=True)
        args = Argument(test_eps=0).parse()
        wrapper = MultiPGDModelWrapper(net, nn.CrossEntropyLoss(), input_dim, device, args)
        return wrapper

    def test_compute_model_stat(self):
        x, y = self._get_random_input()
        wrapper = self._get_model_wrapper_to_test()
        wrapper.max_eps = 0.2
        wrapper.robust_weight = 1
        wrapper.summary_accu_stat = True
        eps = 0.1
        (loss1, nat_loss1, robust_loss1), (nat_accu1, robust_accu1) = wrapper.compute_model_stat(x, y, eps)
        self.assertTrue(has_no_grad(wrapper.net))
        loss1.backward()
        self.assertFalse(has_no_grad(wrapper.net))
        wrapper.grad_cleaner.zero_grad()
        # compare to vanilla PGD
        wrapper = PGDModelWrapper(wrapper.net, nn.CrossEntropyLoss(), wrapper.input_dim, wrapper.device, wrapper.args, enable_EDAC=False)
        wrapper.max_eps = 0.2
        wrapper.robust_weight = 1
        wrapper.summary_accu_stat = True
        (loss2, nat_loss2, robust_loss2), (nat_accu2, robust_accu2) = wrapper.compute_model_stat(x, y, eps)
        self.assertGreater(loss1, loss2)
        self.assertAlmostEqual(nat_loss1.item(), nat_loss2.item(), delta=1e-6)
        self.assertLessEqual(robust_accu1, robust_accu2)
        self.assertEqual(nat_accu1, nat_accu2)
        self.assertTrue(has_no_grad(wrapper.net))

class TestTAPSModelWrapper(TestBoxModelWrapper):
    def _get_model_wrapper_to_test(self):
        device = "cpu"
        net, input_dim, num_class = get_test_network()
        args = Argument(test_eps=0, block_sizes=(3, 4)).parse()
        wrapper = TAPSModelWrapper(net, nn.CrossEntropyLoss(), input_dim, device, args, args.block_sizes)
        return wrapper
    
    def test_property(self):
        super().test_property()
        wrapper = self._get_model_wrapper_to_test()
        # soft thre
        self._test_set_value_type(wrapper, "soft_thre", ("str", "NoneType"), (0.1, 0.2))
        # TAPS grad scale
        self._test_set_value_type(wrapper, "TAPS_grad_scale", ("str", "NoneType"), (0.1, 0.2))
        # no ibp multiplier
        self._test_set_value_type(wrapper, "no_ibp_multiplier", ("str", "numeric", "NoneType"), (False, True))
        # disable TAPS
        self._test_set_value_type(wrapper, "disable_TAPS", ("str", "numeric", "NoneType"), (False, True))
        # latent search restarts
        self._test_set_value_type(wrapper, "latent_search_restarts", ("str", "NoneType"), (1, 2))
        with self.assertRaises(AssertionError):
            wrapper.latent_search_restarts = 0 # must be a positive integer
        # latent search steps
        self._test_set_value_type(wrapper, "latent_search_steps", ("str", "NoneType"), (1, 2))
        with self.assertRaises(AssertionError):
            wrapper.latent_search_steps = 0
        # net blocks abs
        self._test_set_value_type(wrapper, "net_blocks_abs", ("str", "numeric", "NoneType"), ())

    def test_compute_model_stat(self):
        x, y = self._get_random_input()
        wrapper = self._get_model_wrapper_to_test()
        wrapper.max_eps = 0.2
        wrapper.net.train()
        # normal test
        eps = 0.1
        wrapper.robust_weight = 1
        (loss, nat_loss, robust_loss), (nat_accu, robust_accu) = wrapper.compute_model_stat(x, y, eps)
        self.assertEqual(wrapper.current_eps, eps)
        self.assertTrue(has_no_grad(wrapper.net))
        loss.backward()
        self.assertFalse(has_no_grad(wrapper.net))
        wrapper.grad_cleaner.zero_grad()
        # consistency test for eps=0
        eps = 0
        wrapper.no_ibp_multiplier = True
        (loss, nat_loss, robust_loss), (nat_accu, robust_accu) = wrapper.compute_model_stat(x, y, eps)
        self.assertAlmostEqual(nat_loss.item(), robust_loss.item(), delta=1e-6)
        self.assertEqual(nat_accu, robust_accu)
        # test disable TAPS
        wrapper.disable_TAPS = True
        wrapper.freeze_BN = True
        eps = 0.1
        (loss1, nat_loss1, robust_loss1), (nat_accu1, robust_accu1) = wrapper.compute_model_stat(x, y, eps)
        IBP_wrapper = BoxModelWrapper(wrapper.net, nn.CrossEntropyLoss(), wrapper.input_dim, wrapper.device, wrapper.args)
        IBP_wrapper.max_eps = wrapper.max_eps
        IBP_wrapper.robust_weight = 1
        IBP_wrapper.freeze_BN = True
        (loss2, nat_loss2, robust_loss2), (nat_accu2, robust_accu2) = IBP_wrapper.compute_model_stat(x, y, eps)
        self.assertAlmostEqual(loss1.item(), loss2.item(), delta=1e-6)
        self.assertAlmostEqual(nat_loss1.item(), nat_loss2.item(), delta=1e-6)
        self.assertAlmostEqual(robust_loss1.item(), robust_loss2.item(), delta=1e-6)
        self.assertEqual(nat_accu1, nat_accu2)
        self.assertEqual(robust_accu1, robust_accu2)
        self.assertTrue(has_no_grad(wrapper.net))
        # test loss order
        wrapper.disable_TAPS = False
        wrapper.no_ibp_multiplier = True
        eps = 1e-6
        (loss1, nat_loss1, robust_loss1), (nat_accu1, robust_accu1) = wrapper.compute_model_stat(x, y, eps)
        (loss2, nat_loss2, robust_loss2), (nat_accu2, robust_accu2) = IBP_wrapper.compute_model_stat(x, y, eps)
        self.assertLessEqual(nat_loss1, robust_loss1)
        self.assertLessEqual(robust_loss1, robust_loss2)


class TestSmallBoxModelWrapper(TestBoxModelWrapper):
    def _get_model_wrapper_to_test(self):
        device = "cpu"
        net, input_dim, num_class = get_test_network()
        args = Argument(test_eps=0).parse()
        wrapper = SmallBoxModelWrapper(net, nn.CrossEntropyLoss(), input_dim, device, args)
        return wrapper

    def test_property(self):
        super().test_property()
        wrapper = self._get_model_wrapper_to_test()
        # eps shrinkage
        self._test_set_value_type(wrapper, "eps_shrinkage", ("str", "NoneType"), (0.1, 0.2))
        # cache adv xy
        wrapper.cache_advx = True
        self.assertTrue(wrapper.cache_advx)
        with self.assertRaises(Exception):
            wrapper.cache_advx = "a"
        wrapper.cache_advy = True
        self.assertTrue(wrapper.cache_advy)
        with self.assertRaises(Exception):
            wrapper.cache_advy = "a"
        # input search steps
        self._test_set_value_type(wrapper, "input_search_steps", ("str", "NoneType"), (1, 2))
        with self.assertRaises(Exception):
            wrapper.input_search_steps = 0 # must be a positive integer
        # input search restarts
        self._test_set_value_type(wrapper, "input_search_restarts", ("str", "NoneType"), (1, 2))
        with self.assertRaises(Exception):
            wrapper.input_search_restarts = 0 # must be a positive integer
        

    def test_compute_model_stat(self):
        x, y = self._get_random_input()
        wrapper = self._get_model_wrapper_to_test()
        wrapper.max_eps = 0.2
        wrapper.net.train()
        # normal test
        eps = 0.1
        wrapper.robust_weight = 1
        (loss, nat_loss, robust_loss), (nat_accu, robust_accu) = wrapper.compute_model_stat(x, y, eps)
        self.assertEqual(wrapper.current_eps, eps)
        self.assertTrue(has_no_grad(wrapper.net))
        loss.backward()
        self.assertFalse(has_no_grad(wrapper.net))
        wrapper.grad_cleaner.zero_grad()
        # consistency test for eps=0
        eps = 0
        (loss, nat_loss, robust_loss), (nat_accu, robust_accu) = wrapper.compute_model_stat(x, y, eps)
        self.assertAlmostEqual(nat_loss.item(), robust_loss.item(), delta=1e-6)
        self.assertEqual(nat_accu, robust_accu)
        # consistency test for eps_shrinkage=1
        wrapper.eps_shrinkage = 1
        (loss1, nat_loss1, robust_loss1), (nat_accu1, robust_accu1) = wrapper.compute_model_stat(x, y, eps)
        IBP_wrapper = BoxModelWrapper(wrapper.net, nn.CrossEntropyLoss(), wrapper.input_dim, wrapper.device, wrapper.args)
        IBP_wrapper.max_eps = wrapper.max_eps
        IBP_wrapper.robust_weight = 1
        IBP_wrapper.freeze_BN = True
        (loss2, nat_loss2, robust_loss2), (nat_accu2, robust_accu2) = IBP_wrapper.compute_model_stat(x, y, eps)
        self.assertAlmostEqual(loss1.item(), loss2.item(), delta=1e-6)
        self.assertAlmostEqual(nat_loss1.item(), nat_loss2.item(), delta=1e-6)
        self.assertAlmostEqual(robust_loss1.item(), robust_loss2.item(), delta=1e-6)
        self.assertEqual(nat_accu1, nat_accu2)
        self.assertEqual(robust_accu1, robust_accu2)
        self.assertTrue(has_no_grad(wrapper.net))
        # test loss order
        wrapper.eps_shrinkage = 0.5
        wrapper.freeze_BN = True
        eps = 1e-6
        (loss1, nat_loss1, robust_loss1), (nat_accu1, robust_accu1) = wrapper.compute_model_stat(x, y, eps)
        (loss2, nat_loss2, robust_loss2), (nat_accu2, robust_accu2) = IBP_wrapper.compute_model_stat(x, y, eps)
        self.assertLessEqual(nat_loss1, robust_loss1)
        self.assertLessEqual(robust_loss1, robust_loss2)

class TestSTAPSModelWrapper(TestTAPSModelWrapper, TestSmallBoxModelWrapper):
    def _get_model_wrapper_to_test(self):
        device = "cpu"
        net, input_dim, num_class = get_test_network()
        args = Argument(test_eps=0).parse()
        wrapper = STAPSModelWrapper(net, nn.CrossEntropyLoss(), input_dim, device, args, block_sizes=(3, 4), eps_shrinkage=0.5)
        return wrapper
    
    def test_compute_model_stat(self):
        x, y = self._get_random_input()
        wrapper = self._get_model_wrapper_to_test()
        wrapper.max_eps = 0.2
        wrapper.net.train()
        # normal test
        eps = 0.1
        wrapper.robust_weight = 1
        (loss, nat_loss, robust_loss), (nat_accu, robust_accu) = wrapper.compute_model_stat(x, y, eps)
        self.assertEqual(wrapper.current_eps, eps)
        self.assertTrue(has_no_grad(wrapper.net))
        loss.backward()
        self.assertFalse(has_no_grad(wrapper.net))

class TestMTLIBPModelWrapper(TestBoxModelWrapper):
    def _get_model_wrapper_to_test(self):
        device = "cpu"
        net, input_dim, num_class = get_test_network()
        args = Argument(test_eps=0).parse()
        wrapper = MTLIBPModelWrapper(net, nn.CrossEntropyLoss(), input_dim, device, args, ibp_coef=0.5, attack_range_scale=1)
        return wrapper
    
    def test_property(self):
        super().test_property()
        wrapper = self._get_model_wrapper_to_test()
        # ibp coef
        self._test_set_value_type(wrapper, "ibp_coef", ("str", "NoneType"), (0.1, 0.2))
        # attack range scale
        self._test_set_value_type(wrapper, "attack_range_scale", ("str", "NoneType"), (0.1, 0.2))
        # num_steps
        self._test_set_value_type(wrapper, "num_steps", ("str", "NoneType"), (1, 2))
        with self.assertRaises(Exception):
            wrapper.num_steps = 0
        # restarts
        self._test_set_value_type(wrapper, "restarts", ("str", "NoneType"), (1, 2))
        with self.assertRaises(Exception):
            wrapper.restarts = 0
        # step_size
        self._test_set_value_type(wrapper, "step_size", ("str", "NoneType"), (0.1, 0.2))

    def test_compute_model_stat(self):
        x, y = self._get_random_input()
        wrapper = self._get_model_wrapper_to_test()
        wrapper.max_eps = 0.2
        wrapper.net.train()
        # normal test
        eps = 0.1
        wrapper.robust_weight = 1
        (loss, nat_loss, robust_loss), (nat_accu, robust_accu) = wrapper.compute_model_stat(x, y, eps)
        self.assertEqual(wrapper.current_eps, eps)
        self.assertTrue(has_no_grad(wrapper.net))
        loss.backward()
        self.assertFalse(has_no_grad(wrapper.net))
        wrapper.grad_cleaner.zero_grad()
        # consistency test for eps=0
        eps = 0
        (loss, nat_loss, robust_loss), (nat_accu, robust_accu) = wrapper.compute_model_stat(x, y, eps)
        self.assertAlmostEqual(nat_loss.item(), robust_loss.item(), delta=1e-6)
        self.assertEqual(nat_accu, robust_accu)
        # test loss order
        wrapper.freeze_BN = True
        eps = 0.01
        (loss1, nat_loss1, robust_loss1), (nat_accu1, robust_accu1) = wrapper.compute_model_stat(x, y, eps)
        wrapper.attack_range_scale *= 2
        (loss2, nat_loss2, robust_loss2), (nat_accu2, robust_accu2) = wrapper.compute_model_stat(x, y, eps)
        self.assertLess(nat_loss1, robust_loss1)
        self.assertLess(robust_loss1, robust_loss2)
        wrapper.ibp_coef *= 0.5
        (loss3, nat_loss3, robust_loss3), (nat_accu3, robust_accu3) = wrapper.compute_model_stat(x, y, eps)
        self.assertGreater(robust_loss2, robust_loss3)

        
class TestEXPIBPModelWrapper(TestMTLIBPModelWrapper):
    def _get_model_wrapper_to_test(self):
        device = "cpu"
        net, input_dim, num_class = get_test_network()
        args = Argument(test_eps=0).parse()
        wrapper = EXPIBPModelWrapper(net, nn.CrossEntropyLoss(), input_dim, device, args, ibp_coef=0.5, attack_range_scale=2)
        return wrapper
    
class TestCCIBPModelWrapper(TestMTLIBPModelWrapper):
    def _get_model_wrapper_to_test(self):
        device = "cpu"
        net, input_dim, num_class = get_test_network()
        args = Argument(test_eps=0).parse()
        wrapper = CCIBPModelWrapper(net, nn.CrossEntropyLoss(), input_dim, device, args, ibp_coef=0.5, attack_range_scale=2)
        return wrapper

class TestBasicFunctionWrapper(Tester):
    def _get_model_wrapper_to_test(self, modeltype:str="box"):
        device = "cpu"
        net, input_dim, num_class = get_test_network(small=True)
        args = Argument(test_eps=0, grad_accu_batch=2).parse()
        if modeltype == "box":
            wrapper = BoxModelWrapper(net, nn.CrossEntropyLoss(), input_dim, device, args)
        elif modeltype == "pgd":
            wrapper = PGDModelWrapper(net, nn.CrossEntropyLoss(), input_dim, device, args, enable_EDAC=False)
        elif modeltype == "taps":
            wrapper = TAPSModelWrapper(net, nn.CrossEntropyLoss(), input_dim, device, args, block_sizes=(3, 4))
        elif modeltype == "smallbox":
            wrapper = SmallBoxModelWrapper(net, nn.CrossEntropyLoss(), input_dim, device, args, eps_shrinkage=0.5)
        elif modeltype == "staps":
            wrapper = STAPSModelWrapper(net, nn.CrossEntropyLoss(), input_dim, device, args, block_sizes=(3, 4), eps_shrinkage=0.5)
        elif modeltype == "mtlibp":
            wrapper = MTLIBPModelWrapper(net, nn.CrossEntropyLoss(), input_dim, device, args, ibp_coef=0.5, attack_range_scale=1)
        elif modeltype == "expibp":
            wrapper = EXPIBPModelWrapper(net, nn.CrossEntropyLoss(), input_dim, device, args, ibp_coef=0.5, attack_range_scale=1)
        elif modeltype == "ccibp":
            wrapper = CCIBPModelWrapper(net, nn.CrossEntropyLoss(), input_dim, device, args, ibp_coef=0.5, attack_range_scale=1)
        else:
            raise ValueError(f"modeltype {modeltype} not supported")
        wrapper = BasicFunctionWrapper(wrapper)
        return wrapper

    def test_property(self):
        super().test_property()
        wrapper = self._get_model_wrapper_to_test()
        # attribute list
        self._test_set_value_type(wrapper, "attribute_list", ("str", "numeric", "NoneType"), ())
    
class TestGradAccuFunctionWrapper(TestBasicFunctionWrapper):
    def _get_model_wrapper_to_test(self, modeltype:str="box"):
        args = Argument(test_eps=0, grad_accu_batch=2).parse()
        wrapper = super()._get_model_wrapper_to_test(modeltype).wrapper
        wrapper = GradAccuFunctionWrapper(wrapper, args)
        return wrapper

    def test_property(self):
        super().test_property()
        wrapper = self._get_model_wrapper_to_test()
        # accu batch size
        self._test_set_value_type(wrapper, "grad_accu_batch", ("str", "NoneType"), (1, 2))
        # disable accumulation
        self._test_set_value_type(wrapper, "disable_accumulation", ("str", "numeric", "NoneType"), (False, True))

    def test_compute_model_stat(self, modeltype:str="box"):
        x, y = self._get_random_input()
        bs = len(x)
        wrapper = self._get_model_wrapper_to_test(modeltype)
        original_wrapper = wrapper.wrapper
        wrapper.max_eps = 0.2
        # normal test
        eps = 0.1
        wrapper.robust_weight = 1
        (loss, nat_loss, robust_loss), (nat_accu, robust_accu) = wrapper.compute_model_stat(x, y, eps)
        self.assertFalse(has_no_grad(wrapper.net))
        wrapper.grad_cleaner.zero_grad()
        # test disable accumulation
        wrapper.disable_accumulation = True
        (loss, nat_loss, robust_loss), (nat_accu, robust_accu) = wrapper.compute_model_stat(x, y, eps)
        loss.backward()
        named_grads = {}
        for k, p in wrapper.net.named_parameters():
            named_grads[k] = p.grad.clone().detach()
        wrapper.grad_cleaner.zero_grad()
        (loss, nat_loss, robust_loss), (nat_accu, robust_accu) = original_wrapper.compute_model_stat(x, y, eps)
        loss.backward()
        for k, p in wrapper.net.named_parameters():
            self.assertTrue((p.grad == named_grads[k]).all())
        # test accumulation
        wrapper.disable_accumulation = False
        for grad_accu_batch in range(1, bs+1):
            wrapper.grad_cleaner.zero_grad()
            wrapper.grad_accu_batch = grad_accu_batch
            (loss, nat_loss, robust_loss), (nat_accu, robust_accu) = wrapper.compute_model_stat(x, y, eps)
            for k, p in wrapper.net.named_parameters():
                self.assertAlmostEqual((p.grad - named_grads[k]).abs().max().item(), 0, delta=1e-6)
            # test loss.backward does nothing
            loss.backward()
            for k, p in wrapper.net.named_parameters():
                self.assertAlmostEqual((p.grad - named_grads[k]).abs().max().item(), 0, delta=1e-6)
  
class TestWeightSmoothFunctionWrapper(TestBasicFunctionWrapper):
    def _get_model_wrapper_to_test(self, modeltype:str="box"):
        args = Argument(test_eps=0, std_scale=0.1).parse()
        wrapper = super()._get_model_wrapper_to_test(modeltype).wrapper
        wrapper = WeightSmoothFunctionWrapper(wrapper, args.std_scale)
        return wrapper
    
    def test_property(self):
        super().test_property()
        # std scale
        wrapper = self._get_model_wrapper_to_test()
        self._test_set_value_type(wrapper, "std_scale", ("str", "NoneType"), (0.1, 0.2))

    def test_compute_model_stat(self, modeltype:str="box"):
        x, y = self._get_random_input()
        wrapper = self._get_model_wrapper_to_test(modeltype)
        wrapper.current_lr = 1
        original_wrapper = wrapper.wrapper
        layers = [layer for layer in wrapper.net if isinstance(layer, Linear) or isinstance(layer, Conv2d)]
        original_weights = {}
        for id, layer in enumerate(layers):
            original_weights[id] = layer.weight.clone().detach()
        wrapper.max_eps = 0.2
        # normal test
        eps = 0.1
        wrapper.robust_weight = 1
        (loss, nat_loss, robust_loss), (nat_accu, robust_accu) = wrapper.compute_model_stat(x, y, eps)
        self.assertTrue(has_no_grad(wrapper.net))
        # test whether param has been perturbed
        for id, layer in enumerate(layers):
            self.assertNotAlmostEqual((layer.weight - original_weights[id]).abs().max().item(), 0, delta=1e-6)
        # backward
        loss.backward()
        for id, layer in enumerate(layers):
            self.assertNotAlmostEqual((layer.weight - original_weights[id]).abs().max().item(), 0, delta=1e-6)
        wrapper.param_postprocess()
        # test whether param postprocess offsets the perturbation
        for id, layer in enumerate(layers):
            self.assertAlmostEqual((layer.weight - original_weights[id]).abs().max().item(), 0, delta=1e-6)

class TestSAMFunctionWrapper(TestBasicFunctionWrapper):
    def _get_model_wrapper_to_test(self, modeltype: str = "box"):
        wrapper = super()._get_model_wrapper_to_test(modeltype).wrapper
        wrapper = SAMFunctionWrapper(wrapper, rho=0.1, adaptive_rho=False)
        return wrapper
    
    def test_property(self):
        super().test_property()
        # rho
        wrapper = self._get_model_wrapper_to_test()
        self._test_set_value_type(wrapper, "rho", ("str", "NoneType"), (0.1, 0.2))
        # adaptive rho
        self._test_set_value_type(wrapper, "adaptive_rho", ("str", "numeric", "NoneType"), (False, True))

    def test_compute_model_stat(self, modeltype:str="box"):
        x, y = self._get_random_input()
        wrapper = self._get_model_wrapper_to_test(modeltype)
        wrapper.current_lr = 1
        original_wrapper = wrapper.wrapper
        layers = [layer for layer in wrapper.net if isinstance(layer, Linear) or isinstance(layer, Conv2d)]
        original_weights = {}
        for id, layer in enumerate(layers):
            original_weights[id] = layer.weight.clone().detach()
        wrapper.max_eps = 0.2
        # normal test
        eps = 0.1
        wrapper.robust_weight = 1
        (loss, nat_loss, robust_loss), (nat_accu, robust_accu) = wrapper.compute_model_stat(x, y, eps)
        self.assertTrue(has_no_grad(wrapper.net))
        # test whether param has been perturbed
        for id, layer in enumerate(layers):
            self.assertNotAlmostEqual((layer.weight - original_weights[id]).abs().max().item(), 0, delta=1e-6)
        # backward
        loss.backward()
        for id, layer in enumerate(layers):
            self.assertNotAlmostEqual((layer.weight - original_weights[id]).abs().max().item(), 0, delta=1e-6)
        wrapper.param_postprocess()
        # test whether param postprocess offsets the perturbation
        for id, layer in enumerate(layers):
            self.assertAlmostEqual((layer.weight - original_weights[id]).abs().max().item(), 0, delta=1e-6)


if __name__ == "__main__":
    seed_everything(123)
    unittest.main()