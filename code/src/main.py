import torch

import utility
import data
import model
import loss
from option import args
from trainer import Trainer
import torch.backends.cudnn as cudnn


def network_parameters(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print('Total number of parameters: %d' % num_params)


def print_setting(net, args):
     print('init this train:')
     network_parameters(net)
     print('training model:', args.model)
     print('scale:', args.scale)
     print('resume from ', args.resume)
     print('output patch size', args.patch_size)
     print('optimization setting: ', args.optimizer)
     print('total epochs:', args.epochs)
     print('lr:', args.lr, 'lr_decay at:', args.decay_type, 'decay gamma:', args.gamma)
     print('train loss:', args.loss)
     print('save_name:', args.save)


torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)
if checkpoint.ok:
    loader = data.Data(args)
    model = model.Model(args, checkpoint)
    print_setting(model, args)
    loss = loss.Loss(args, checkpoint) if not args.test_only else None
    t = Trainer(args, loader, model, loss, checkpoint)
    while not t.terminate():
        cudnn.benchmark = args.cudnn
        t.train()
        t.test()

    checkpoint.done()

