import torch
import argparse
import learner as ln
from GO_data import GOData
from utils import substeps, l1

parser = argparse.ArgumentParser()
parser.add_argument('--device',type=int, default=0)
parser.add_argument('--i',type=int, default=1)
parser.add_argument('--s',type=int, default=1)
args = parser.parse_args()
def main():
    
    run(index=args.i, Nintegrator = 'AB', m=2, seed=args.s)
    run(index=args.i, Nintegrator = 'BDF', m=2, seed=args.s)
    run(index=args.i, Nintegrator = 'AM', m=2, seed=args.s)

    run(index=args.i, Nintegrator = 'AB', m=3, seed=args.s)
    run(index=args.i, Nintegrator = 'BDF', m=3, seed=args.s)
    run(index=args.i, Nintegrator = 'AM', m=3, seed=args.s)

    run(index=args.i, Nintegrator = 'AB', m=1, seed=args.s)
    run(index=args.i, Nintegrator = 'BDF', m=1, seed=args.s)

def run(index=2, Nintegrator = 'AB', m=2, seed=0):
    if torch.cuda.is_available():
        device = 'gpu'
        torch.cuda.set_device(args.device)
    else: 
        device ='cpu'
    
    Nlayers =3
    Nwidth =128
    Nactivation = 'tanh'
   
    h=0.002*index
    train_traj_num = 60
    test_traj_num=40
    data = GOData(train_traj_num=train_traj_num, test_traj_num=test_traj_num, 
                  substeps=substeps[Nintegrator][str(m)], index=index)
    

    net = ln.nn.LMNets(dim=data.dim, layers=Nlayers, width=Nwidth, activation=Nactivation, 
                 integrator=Nintegrator, m=m, h=h)
    filename = 'GO_'+Nintegrator+'_{}_{}_{}'.format(m, index, seed)
    batch_size=30000 if index <10 else None
    arguments = {
        'filename': filename,
        'data': data,
        'net': net,
        'criterion': None,
        'optimizer': 'adam',
        'lr': 0.01,
        'lr_decay': 100,
        'iterations': 300000,
        'batch_size': batch_size,
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
    a=np.zeros([2])
    net = torch.load('./outputs/'+filename + '/model_best.pkl')
    data.device=device
    data.dtype='float'
    a[0] = l1(data.f(data.X_test_np[0])-net.vf(data.X_test[0]).cpu().detach().numpy())
    a[1] = l1((torch.sum(net.a*data.X_test, dim=0)/net.h-torch.sum(net.b*net.vf(data.X_test), dim=0)).cpu().detach().numpy())
    np.savetxt('./outputs/'+filename+'/error.txt', a)
    
if __name__ == '__main__':
    main()