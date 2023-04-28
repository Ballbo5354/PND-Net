import torch
import torch.nn as nn


def get_state_dict_on_cpu(obj):
    cpu_device = torch.device('cpu')
    state_dict = obj.state_dict()
    for key in state_dict.keys():
        state_dict[key] = state_dict[key].to(cpu_device)
    return state_dict


def save_ckpt(ckpt_name, models, optimizers, n_iter):
    ckpt_dict = {'n_iter': n_iter}
    for prefix, model in models:
        ckpt_dict[prefix] = get_state_dict_on_cpu(model)

    if optimizers:
        for prefix, optimizer in optimizers:
            ckpt_dict[prefix] = optimizer.state_dict()
    torch.save(ckpt_dict, ckpt_name)


def load_ckpt(ckpt_name, models, optimizers=None):
    ckpt_dict = torch.load(ckpt_name)
    for prefix, model in models:
        assert isinstance(model, nn.Module)
        model.load_state_dict(ckpt_dict[prefix], strict=False)
    if optimizers is not None:
        for prefix, optimizer in optimizers:
            optimizer.load_state_dict(ckpt_dict[prefix])
    return ckpt_dict['n_iter']


class LambdaLR():
    def __init__(self, n_epochs, offset, decay_strat_epoch):
        assert ((n_epochs - decay_strat_epoch)>0),"Decay must strat before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_strat_epoch = decay_strat_epoch

    def step(self,epoch):
        return 1.0 - max(0, epoch+self.offset-self.decay_strat_epoch)/(self.n_epochs-self.decay_strat_epoch)