import torch
import argparse

import learner as ln
from DO_data import DOData, DOIMDE
from utils import substeps, l1

parser = argparse.ArgumentParser()
parser.add_argument('--device',type=int, default=1)
parser.add_argument('--i',type=int, default=1)
parser.add_argument('--s',type=int, default=1)

args = parser.parse_args()
def main():

    run(h=0.002*args.i, Nintegrator = 'AB', m=1, seed=args.s)
    run(h=0.002*args.i, Nintegrator = 'AB', m=2, seed=args.s)
    
    run(h=0.002*args.i, Nintegrator = 'BDF', m=1, seed=args.s)
    run(h=0.002*args.i, Nintegrator = 'BDF', m=2, seed=args.s) 
    
    run(h=0.002*args.i, Nintegrator = 'AM', m=2, seed=args.s)
    run(h=0.002*args.i, Nintegrator = 'AM', m=3, seed=args.s)
    
def run(h=0.01, Nintegrator = 'AB', m=1, seed=1):
    if torch.cuda.is_available():
        device = 'gpu'
        torch.cuda.set_device(args.device)
    else: 
        device ='cpu'    
    print(Nintegrator, m)
    Nlayers =3
    Nwidth =128
    Nactivation = 'tanh'
   
    h=h
    train_traj_num = 1500
    test_traj_num=500
    
    data = DOData(train_traj_num=train_traj_num, test_traj_num=test_traj_num, length =10, 
                  h=h, substeps=substeps[Nintegrator][str(m)])
    

    net = ln.nn.LMNets(dim=data.dim, layers=Nlayers, width=Nwidth, activation=Nactivation, 
                 integrator=Nintegrator, m=m, h=h)
    
    filename='DO_'+Nintegrator+'_{}_{}_{}'.format(m,h,seed)
    arguments = {
        'filename': filename,
        'data': data,
        'net': net,
        'criterion': None,
        'optimizer': 'adam',
        'lr': 0.01,
        'lr_decay': 100,
        'iterations': 100000,
        'batch_size': None,
        'print_every': 1000,
        'save': True,
        'callback': None,
        'dtype': 'float',
        'device': device
    }
  

    ln.Brain.Init(**arguments)
    ln.Brain.Run()
    ln.Brain.Restore()
    ln.Brain.Output()
    
    import numpy as np
    a=np.zeros([3])
    net = torch.load('./outputs/'+filename + '/model_best.pkl')
    data.device=device
    data.dtype='float'
    print(data.device)
    print(net.vf(data.X_test[0]).shape)
    print(l1(DOIMDE(integrator=Nintegrator, m=m, h=h).F(data.X_test_np[0])-net.vf(data.X_test[0]).cpu().detach().numpy()))
    a[0] = l1(data.f(data.X_test_np[0])-net.vf(data.X_test[0]).cpu().detach().numpy())
    a[1] = l1(DOIMDE(integrator=Nintegrator, m=m, h=h).F(data.X_test_np[0])-net.vf(data.X_test[0]).cpu().detach().numpy())
    a[2] = l1((torch.sum(net.a*data.X_test, dim=0)/net.h-torch.sum(net.b*net.vf(data.X_test), dim=0)).cpu().detach().numpy())
    
    print(a)
    np.savetxt('./outputs/'+filename+'/error.txt', a)


if __name__ == '__main__':
    main()
