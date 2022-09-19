import numpy as np
import matplotlib.pyplot as plt
import torch


from DO_data import DOData, DOIMDE
from utils import l1


xsize=27
legendsize=27
ticksize=24
titlesize=29
linewidth=3
hlinewidth = 4


def plot_traj():
    fig, ax=plt.subplots(2,3, figsize=(28,14))
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None,
            wspace=0.25, hspace=0.3)
    
    
    h=0.01
    steps = 1000
    x0=[2.,0.]
    data= DOData(train_traj_num=1, test_traj_num=1)
    t = np.linspace(0, h*steps, steps+1)
    flow_true = data.solver.flow(np.array(x0), h, steps)

    for ax1 in ax:
        for a in ax1:
            a.plot(t, flow_true[0:, 0], color='black', label='Exact ODE', linewidth= linewidth, zorder=0) 
    
    plot_traj_inte(ax[0,0], integrator='AB', m=1, steps = steps)
    plot_traj_inte(ax[1,0], integrator='AB', m=2, steps = steps)
    plot_traj_inte(ax[0,1], integrator='BDF', m=1, steps = steps)
    plot_traj_inte(ax[1,1], integrator='BDF', m=2, steps = steps)
    plot_traj_inte(ax[0,2], integrator='AM', m=2, steps = steps)
    plot_traj_inte(ax[1,2], integrator='AM', m=3, steps = steps)
    
    ax[1,1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), 
                  fontsize=legendsize, frameon=False, ncol=3)
    
    fig.savefig('DO_traj.pdf', bbox_inches='tight')


def plot_traj_inte(ax, integrator='AB', m=1, steps = 10):
    H=0.01
    h=0.01
    x0=[2.,0.]
    t = np.linspace(0, h*steps, steps+1)
    local = '.\\DOoutputs\\DO_'+integrator+'_{}_{}_0\\model_best.pkl'.format(m, H)
    net = torch.load(local, map_location='cpu')
    flow_pred = net.predict(torch.tensor(x0), h, steps, keepinitx=True, returnnp=True)
    flow_modi = DOIMDE(integrator=integrator, m=m, h=H).solver.flow(np.array(x0), h, steps)
    
    ax.plot(t, flow_modi[0:, 0], color='b', label='IMDE', linewidth= linewidth, zorder=1)
    ax.plot(t, flow_pred[0:, 0], color='red', label='LMNets', linestyle='--', dashes=(4,4), linewidth= linewidth, zorder=2)
    
    if integrator=='AM':
        ax.set_title(integrator+'{}'.format(m-1),fontsize=titlesize,loc='center')
    else:
        ax.set_title(integrator+'{}'.format(m), fontsize=titlesize, loc= 'center')     
    ax.set_xlim(-0.5,11.5)
    ax.set_ylim(-2.2,2.2)
    ax.set_xlabel(r'$T$', fontsize=xsize, loc='right')
    ax.set_ylabel(r'$p$', fontsize=xsize+1)
    ax.tick_params(labelsize=ticksize)


def plot_error_inte1(ax, data, integrator='AB', m=1):
    
    d=5
    a=np.zeros([3,8])#mean
    b=np.zeros([3,8])#std
    
    e_f=np.zeros([d,8])
    e_m=np.zeros([d,8])
    e_l=np.zeros([d,8])
    H=[]
    for i in range(1,9):
        H.append(0.002*i)
    for i in range(1, 9):
        for j in range(d):
            local = '.\\DOoutputs\\DO_'+integrator+'_{}_{}_{}'.format(m, 0.002*i, j)
            net = torch.load(local + '\\model_best.pkl', map_location='cpu')
            e_f[j, i-1] = l1(data.f(data.X_test_np[0])-net.vf(data.X_test[0]).detach().numpy())
            e_m[j, i-1] = l1(DOIMDE(integrator=integrator, m=m, h=0.002*i).F(data.X_test_np[0])-net.vf(data.X_test[0]).detach().numpy())
            error=np.loadtxt(local+'\\error.txt')
            e_l[j, i-1] = error[2]
    a[0,:] = e_f.mean(0)
    a[1,:] = e_m.mean(0)
    a[2,:] = e_l.mean(0)
    
    b[0,:] = e_f.std(0)
    b[1,:] = e_m.std(0)
    b[2,:] = e_l.std(0)
    
    
    
    ax.set_xlim(0.0005, 0.018)
    
    ax.set_yscale('log')
    ax.set_xscale('log')
    
    
    c=a[0,-1]/0.016**m
    H_y=[]
    for h in H:
        H_y.append(c*h**m)
        la=['1st', '2nd', '3rd']
    
    ax.plot(H,a[0],'o--', c = 'red', zorder=2, label=r"Error($f_{\theta}$, $f$ )",linewidth = hlinewidth, markersize=10) 
    ax.plot(H,a[1],'>--', c = 'blue', zorder=1, label=r"Error($f_{\theta}$, $f_h$)",linewidth = hlinewidth, markersize=10)
    ax.plot(H,a[2],'<--', c = 'orange', zorder=0, label = 'Test Loss',linewidth = hlinewidth, markersize=10)     
    ax.plot(H,H_y,'-', c = 'gray', zorder=1,label=la[m-1], linewidth = hlinewidth) 
    
    ax.fill_between(H,a[0]-b[0], a[0]+b[0], alpha=0.2, facecolor = 'red', zorder=2)     
    ax.fill_between(H,a[1]-b[1], a[1]+b[1], alpha=0.2, facecolor = 'blue', zorder=1) 
    ax.fill_between(H,a[2]-b[2], a[2]+b[2], alpha=0.2, facecolor = 'orange', zorder=0) 
        
    if integrator=='AM':
        ax.set_ylim(0.001, 0.04)
        ax.set_title(integrator+'{}'.format(m-1),fontsize=titlesize,loc='center')
    else:
        ax.set_ylim(0.0005, 0.4)
        ax.set_title(integrator+'{}'.format(m),fontsize=titlesize,loc='center')
    ax.set_xlabel(r'$h$', fontsize=xsize+1, loc='right')
    ax.set_ylabel(r'Error', fontsize=xsize+1)
    ax.tick_params(labelsize=ticksize) 
    ax.legend(loc='upper left', fontsize=legendsize)
def plot_error1():
    fig, ax=plt.subplots(3,2, figsize=(28,24))
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None,
            wspace=0.2, hspace=0.35)
    data =  DOData(train_traj_num=1, test_traj_num=10000, length =0, 
                  h=0, substeps=1, region = [[-2, 2], [-2, 2]])
    print(data.X_test.shape)
    data.device='cpu'
    data.dtype='float'
    
    plot_error_inte1(ax[0,0], data, integrator='AB', m=1)
    plot_error_inte1(ax[0,1], data, integrator='AB', m=2)
        
    plot_error_inte1(ax[1,0], data, integrator='BDF', m=1)
    plot_error_inte1(ax[1,1], data, integrator='BDF', m=2)        
    
    plot_error_inte1(ax[2,0], data, integrator='AM', m=2)
    plot_error_inte1(ax[2,1], data, integrator='AM', m=3)

    
    fig.savefig('DO_error.pdf', bbox_inches='tight')                 
 
    return 0


if __name__=='__main__':

    
    plot_traj()
    plot_error1()
