import numpy as np
from scipy.special import digamma
import torch
from tqdm import trange
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Function, Variable

_epsilon = 1e-6

def check_dicts(inputs, outputs):
    inlens = np.unique(np.array([len(x) for x in inputs.values()]))
    assert inlens.size == 1, 'Not all iterables are of equal length'

    outlens = np.unique(np.array([len(x) for x in outputs.values()]))
    assert outlens.size == 1, 'Not all iterables are of equal length'

    assert inlens[0] == outlens[0], 'Input output length do not match'


class DictTensorDataset(Dataset):
    def __init__(self, inputs, outputs):
        check_dicts(inputs, outputs)
        inlens = np.unique(np.array([len(x) for x in inputs.values()]))

        self.inputs =  {k: torch.from_numpy(v)
                        if type(v).__name__ == 'ndarray' else v for k, v in inputs.items()}

        self.outputs = {k: torch.from_numpy(v)
                        if type(v).__name__ == 'ndarray' else v for k, v in outputs.items()}
        self.length = inlens[0]

    def __getitem__(self, index):
        return ({k: v[index] for k, v in self.inputs.items()},
                {k: v[index] for k, v in self.outputs.items()})

    def __len__(self):
        return self.length

    def type(self, dtype):
        return DictTensorDataset({k: v.type(dtype) for k, v in self.inputs.items()},
                                 {k: v.type(dtype) for k, v in self.outputs.items()})

    def cuda(self):
        return DictTensorDataset({k: v.cuda() for k, v in self.inputs.items()},
                                 {k: v.cuda() for k, v in self.outputs.items()})

    def float(self):
        return DictTensorDataset({k: v.float() for k, v in self.inputs.items()},
                                 {k: v.float() for k, v in self.outputs.items()})
    def double(self):
        return DictTensorDataset({k: v.double() for k, v in self.inputs.items()},
                                 {k: v.double() for k, v in self.outputs.items()})

class ExpModule(torch.nn.Module):
    def __init__(self, eps=1e6):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        return torch.exp(x).clamp(max=self.eps)


class Lgamma(Function):
    @staticmethod
    def forward(self, input):
        self.save_for_backward(input)
        return torch.lgamma(input)

    @staticmethod
    def backward(self, grad_output):
        input, = self.saved_tensors

        res = Variable(torch.from_numpy(digamma(input.cpu().numpy())).type_as(input))
        return grad_output*res
lgamma = Lgamma.apply

 # log gamma code from pyro:
 # https://github.com/uber/pyro/blob/dev/pyro/distributions/util.py
def lgamma2(xx):
    gamma_coeff = (
        76.18009172947146,
        -86.50532032941677,
        24.01409824083091,
        -1.231739572450155,
        0.1208650973866179e-2,
        -0.5395239384953e-5,
    )
    magic1 = 1.000000000190015
    magic2 = 2.5066282746310005
    x = xx - 1.0
    t = x + 5.5
    t = t - (x + 0.5) * torch.log(t)
    ser = torch.ones_like(x) * magic1
    for c in gamma_coeff:
        x = x + 1.0
        ser = ser + torch.pow(x / c, -1)
    return torch.log(ser * magic2) - t


def lgamma3(z):
    gamma_r10 = 10.900511

    pi = torch.zeros_like(z)
    pi.fill_(np.pi)
    gamma_c = 2.0 * torch.sqrt(np.e/pi)

    #if z < 0:
    #  return torch.log(pi) - torch.log(torch.abs(torch.sin(pi*z))) - lgamma(1.0 - z)

    sum = 2.48574089138753565546e-5
    sum += 1.05142378581721974210  / z
    sum += -3.45687097222016235469 / (z + 1.0)
    sum += 4.51227709466894823700  / (z + 2.0)
    sum += -2.98285225323576655721 / (z + 3.0)
    sum += 1.05639711577126713077  / (z + 4.0)
    sum += -1.95428773191645869583e-1 / (z + 5.0)
    sum += 1.70970543404441224307e-2  / (z + 6.0)
    sum += -5.71926117404305781283e-4 / (z + 7.0)
    sum += 4.63399473359905636708e-6  / (z + 8.0)
    sum += -2.71994908488607703910e-9 / (z + 9.0)

    # For z >= 0 gamma function is positive, no abs() required.
    return torch.log(gamma_c) + (z - 0.5)*torch.log(z  + gamma_r10 - 0.5) - (z - 0.5) + torch.log(sum)



class AdaptiveZINBLoss(torch.nn.Module):
    def __init__(self, theta_shape=None, pi_ridge=0.0, adapt_lambda=0.1, 
                 min_pi=0.01, max_pi=0.99, eps=1e-10):
        super().__init__()
        self.pi_ridge = pi_ridge
        self.adapt_lambda = adapt_lambda
        self.min_pi = min_pi
        self.max_pi = max_pi
        self.eps = eps
        
        self.register_buffer('zero_ratio', torch.tensor(0.5))
        self.register_buffer('update_step', torch.tensor(0))
        
        if theta_shape is not None:
            theta = torch.rand(*theta_shape) * 0.1 + 1.0 
            self.theta = torch.nn.Parameter(torch.log(theta))

    def forward(self, mean, pi, target, theta=None):
        pi = pi.clamp(min=self.min_pi, max=self.max_pi)
        
        if theta is None:
            theta = torch.exp(self.theta).clamp(max=1e6) + self.eps
        
        nb_case = self.nb(mean, target, theta) - torch.log(1.0 - pi + self.eps)
        zero_nb = torch.exp(theta * (torch.log(theta + self.eps) - 
                                   torch.log(theta + mean + self.eps)))
        zero_case = -torch.log(pi + (1.0 - pi) * zero_nb + self.eps)
        
        with torch.no_grad():
            current_zero_ratio = (target == 0).float().mean()
            self.zero_ratio = (1 - self.adapt_lambda) * self.zero_ratio + \
                             self.adapt_lambda * current_zero_ratio
            self.update_step += 1
            
        adapt_weight = torch.sigmoid((self.zero_ratio - 0.5) * 10)
        
        zero_mask = (target == 0.0).float()
        nb_mask = 1.0 - zero_mask
        
        result = (adapt_weight * zero_mask * zero_case + 
                 (1 - adapt_weight) * nb_mask * nb_case)
        
        if self.pi_ridge > 0:
            ridge = self.pi_ridge * torch.where(pi > 0.5, pi, 1-pi).pow(2)
            result = result + ridge
            
        return result.mean()

    def nb(self, mean, target, theta):
        log_theta = torch.log(theta + self.eps)
        log_mean = torch.log(mean + self.eps)
        log_theta_mean = torch.log(theta + mean + self.eps)
        
        t1 = -lgamma(target + theta + self.eps)
        t2 = lgamma(theta + self.eps)
        t3 = lgamma(target + 1.0)
        t4 = -theta * log_theta
        t5 = -target * log_mean
        t6 = (theta + target) * log_theta_mean
        
        return t1 + t2 + t3 + t4 + t5 + t6

    def zero_memberships(self, mean, pi, target, theta=None):
        pi = pi.clamp(min=self.min_pi, max=self.max_pi)
        
        if theta is None:
            theta = torch.exp(self.theta).clamp(max=1e6) + self.eps
            
        log_ratio = theta * (torch.log(theta + self.eps) - 
                           torch.log(theta + mean + self.eps))
        nb_zero_prob = torch.exp(log_ratio)
        
        memberships = pi / (pi + (1 - pi) * nb_zero_prob + self.eps)
        memberships = torch.where(target != 0, torch.zeros_like(memberships), memberships)
        
        return memberships



def train(model_dict, loss_dict, model, loss, optimizer, epochs=1,
          val_split=0.1, val_data=None, batch_size=32, grad_clip=5.0,
          shuffle=True, verbose=0, early_stopping=None, scheduler=None,
          dtype='float'):

    check_dicts(model_dict, loss_dict)
    dataset = DictTensorDataset(model_dict, loss_dict)

    if dtype == 'cuda':
        dataset = dataset.float().cuda()
        model = model.float().cuda()
        loss = loss.float().cuda()
    elif dtype == 'double':
        dataset = dataset.double()
        model = model.double()
        loss = loss.double()
    elif dtype == 'float':
        dataset = dataset.float()
        model = model.float()
        loss = loss.float()
    else:
        raise 'Unknown dtype'

    if shuffle:
        # shuffle dataset
        idx = torch.randperm(len(dataset))
        if dtype == 'cuda':
            idx = idx.cuda()
        dataset = DictTensorDataset(*dataset[idx])

    if val_data is not None:
        train_data = dataset
        val_loader = DataLoader(val_data, batch_size=len(val_data), shuffle=False)
    elif val_split > 0.:
        off = int(len(dataset)*(1.0-val_split))
        train_data = DictTensorDataset(*dataset[:off])
        val_data = DictTensorDataset(*dataset[off:])
        val_loader = DataLoader(val_data, batch_size=len(val_data), shuffle=False)
    else:
        train_data, val_data = dataset, None

    loader = DataLoader(train_data, batch_size=batch_size, shuffle=shuffle)
    result = {'loss': [], 'model': model, 'early_stop': False}

    if verbose:
        it = trange(epochs)
    else:
        it = range(epochs)

    for epoch in it:
        train_batch_losses = []

        for modeld, lossd in loader:
            cur_batch_size = len(lossd['target'])
            modeld = {k: Variable(v) for k, v in modeld.items()}
            lossd  = {k: Variable(v) for k, v in lossd.items()}

            def closure():
                optimizer.zero_grad()
                pred = model(**modeld)
                if not isinstance(pred, dict): pred = {'input': pred}
                l = loss(**pred, **lossd)
                train_batch_losses.append(l.data.cpu().numpy()[0]*cur_batch_size)
                l.backward()
                if grad_clip:
                    for pg in optimizer.param_groups:
                        torch.nn.utils.clip_grad_norm(pg['params'], grad_clip)
                return l

            optimizer.step(closure)

        result['loss'].append(np.array(train_batch_losses).sum()/len(train_data))

        if val_data:
            for modeld, lossd in val_loader:
                modeld = {k: Variable(v) for k, v in modeld.items()}
                lossd  = {k: Variable(v) for k, v in lossd.items()}
                model.eval()
                pred = model(**modeld)
                if not isinstance(pred, dict): pred = {'input': pred}
                l = loss(**pred, **lossd)
                model.train()
                result.setdefault('val_loss', []).append(l.data.cpu().numpy()[0])

        if verbose:
            text = 'Epoch: %s training loss: %s val loss: %s' % ((epoch+1), result['loss'][-1],
                    result['val_loss'][-1] if 'val_loss' in result else  '---')
            print(text)

        if scheduler:
            if val_data is not None:
                scheduler.step(result['val_loss'][-1])
            else:
                if epoch == 0 and scheduler.verbose:
                    print('Validation data not specified, using training loss for lr scheduling')
                scheduler.step(result['loss'][-1])

        if early_stopping and not early_stopping.step(result):
            result['early_stop'] = True
            return result

    return result



def train_em(model_dict, loss_dict, model, loss,
             optimizer, epochs=1, m_epochs=1, val_split=0.1, grad_clip=5.0,
             batch_size=32, shuffle=True, verbose=0, early_stopping=None,
             scheduler=None, dtype='float'):

    memberships = torch.from_numpy(np.zeros_like(loss_dict['target']))
    loss_dict['zero_memberships'] = memberships
    check_dicts(model_dict, loss_dict)
    dataset = DictTensorDataset(model_dict, loss_dict)

    if dtype == 'cuda':
        dataset = dataset.float().cuda()
        model = model.float().cuda()
        loss = loss.float().cuda()
    elif dtype == 'double':
        dataset = dataset.double()
        model = model.double()
        loss = loss.double()
    elif dtype == 'float':
        dataset = dataset.float()
        model = model.float()
        loss = loss.float()
    else:
        raise 'Unknown dtype'

    if shuffle:
        idx = torch.randperm(len(dataset))
        if dtype == 'cuda':
            idx = idx.cuda()
        dataset = DictTensorDataset(*dataset[idx])

    if val_split > 0.:
        off = int(len(dataset)*(1.0-val_split))
        train_data = DictTensorDataset(*dataset[:off])
        val_data = DictTensorDataset(*dataset[off:])
    else:
        train_data, val_data = dataset, None

    ret = {'loss': []}
    if verbose:
        it = trange(int(np.ceil(epochs/m_epochs)))
    else:
        it = range(int(np.ceil(epochs/m_epochs)))

    for i in it:
        train_ret = train(model_dict=train_data.inputs, loss_dict=train_data.outputs,
                          model=model, loss=loss, optimizer=optimizer,
                          epochs=m_epochs, shuffle=shuffle, verbose=0,
                          batch_size=batch_size, grad_clip=grad_clip, val_data=val_data,
                          val_split=0.0, early_stopping=early_stopping,
                          dtype=dtype)
        ret['loss'] += train_ret['loss']

        if val_data:
            ret.setdefault('val_loss', []).extend(train_ret['val_loss'])

        if verbose:
            text = 'Epoch: %s training loss: %s val loss: %s' % ((i+1), ret['loss'][-1], ret['val_loss'][-1] if 'val_loss' in ret else  '---')
            # it.set_description() can also be used but it's better to see errors flowing
            print(text)

        model.eval()
        pred = model(**{k: Variable(v) for k, v in train_data.inputs.items()}) #we need variables here
        memberships = loss.zero_memberships(**pred, target=Variable(train_data.outputs['target'])).data
        train_data.outputs['zero_memberships'] = memberships.clone()

        if val_data is not None:
            pred = model(**{k: Variable(v) for k, v in val_data.inputs.items()})
            memberships = loss.zero_memberships(**pred,
                                                target=Variable(val_data.outputs['target'])).data
            val_data.outputs['zero_memberships'] = memberships.clone()
        model.train()

        if scheduler:
            if val_data is not None:
                scheduler.step(ret['val_loss'][-1])
            else:
                if epochs == 0 and scheduler.verbose:
                    print('Validation data not specified, using training loss for lr scheduling')
                scheduler.step(ret['loss'][-1])

        if train_ret['early_stop']:
            break

    ret['model'] = model

    return ret