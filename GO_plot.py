import numpy as np
import matplotlib.pyplot as plt
import torch

from GO_data import GOData

xsize=25
legendsize=28
ticksize=24
titlesize=27

linewidth = 3


def plot_error_inte_m(ax, integrator='AB', m=1):
    
    d=3
    a=np.zeros([2,5])#mean
    b=np.zeros([2,5])#std
    
    e_f=np.zeros([d,5])
    e_l=np.zeros([d,5])
    H=[]
    for i in range(5):
        H.append(0.002*2**i)
    for i in range(5):
        for j in range(d):
            local = '.\\GOoutputs\\GO_'+integrator+'_{}_{}_{}'.format(m, 2**i, j)
            error=np.loadtxt(local+'\\error.txt')            
            e_f[j, i] = error[0]
            e_l[j, i] = error[1]

    a[0,:] = e_f.mean(0)
    a[1,:] = e_l.mean(0)
    print(integrator, m)
    print(a)
    print(e_f)
    
    b[0,:] = e_f.std(0)
    b[1,:] = e_l.std(0)
    
    c=a[0,-1]/0.032**m
    H_y=[]
    for h in H:
        H_y.append(c*h**m)
        la=['1st', '2nd', '3rd']
    color=['red', 'green', 'blue']
    if integrator=='AM':
        ax.plot(H,a[0],'o-', c = color[m-1], zorder=2, markersize=10,
                label=integrator+str(m-1),linewidth = linewidth)
    else:
        ax.plot(H,a[0],'o-', c = color[m-1], zorder=2, markersize=10,
        label=integrator+str(m),linewidth = linewidth) 

    ax.plot(H,a[1],'<--', c = color[m-1], zorder=0, markersize=10,
            linewidth = linewidth)     
    ax.plot(H,H_y,':', c = color[m-1], zorder=1,label=la[m-1], linewidth = linewidth-1) 
    
    ax.fill_between(H,a[0]-b[0], a[0]+b[0], alpha=0.5, facecolor = color[m-1], zorder=2)     
    ax.fill_between(H,a[1]-b[1], a[1]+b[1], alpha=0.5, facecolor = color[m-1], zorder=1) 

def plot_error_inte(ax, integrator='AB'):
    order = [0,2,1,3]
    if integrator != 'AM':
        order = [0,2,4,1,3,5]
        plot_error_inte_m(ax, integrator=integrator, m=1)
    
    plot_error_inte_m(ax, integrator=integrator, m=2)
    plot_error_inte_m(ax, integrator=integrator, m=3)    
    
    ax.set_xlim(0.00099, 0.04)
    ax.set_ylim(0.0011, 55)
    
    ax.set_yscale('log')
    ax.set_xscale('log')   
    ax.set_title(integrator,fontsize=titlesize,loc='center')
    ax.set_xlabel(r'$h$', fontsize=xsize+1, loc='right')
    ax.set_ylabel('Error', fontsize=xsize+1)
    ax.tick_params(labelsize=ticksize) 
    handles, labels = ax.get_legend_handles_labels()
    
    ax.legend([handles[idx] for idx in order],[labels[idx] for idx in order], loc='upper left', fontsize=legendsize,ncol=2)
  
    
def plot_error():
    fig, ax=plt.subplots(1,3, figsize=(28,10))
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None,
            wspace=0.25, hspace=0.35)
    
    plot_error_inte(ax[0], integrator='AB')
    plot_error_inte(ax[1], integrator='BDF')
    plot_error_inte(ax[2], integrator='AM')
    fig.savefig('GO_error.pdf', bbox_inches='tight')                 
 
    return 0


def plot_traj():
    fig, ax=plt.subplots(3,1, figsize=(25,15))
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None,
            wspace=0.2, hspace=0.4)
    
    
    h=0.002
    steps = 5000

    data= GOData()
    data.device='cpu'
    data.dtype='float'
    x0=data.X_test[0, 0].tolist()#in our experiment, x0 = [0.78262879, 0.7994336,  0.08400743, 0.27925725, 0.20148105, 1.90529823, 0.08089575]
    t = np.linspace(0, h*steps, steps+1)
    flow_true = data.solver.flow(np.array(x0), h, steps)

    for a in ax:
        a.plot(t, flow_true[:, 0], color='black', label='Exact ODE', linewidth= linewidth, zorder=0) 
    
    plot_traj_inte(x0, data, ax[0], integrator='AB', m=2, seed=0,  steps = steps)
    plot_traj_inte(x0, data, ax[1], integrator='BDF', m=2, seed=1, steps = steps)
    plot_traj_inte(x0, data, ax[2], integrator='AM', m=2, seed=0, steps = steps)

    ax[2].legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), 
                  fontsize=legendsize, frameon=False, ncol=2)
    
    fig.savefig('GO_traj.pdf', bbox_inches='tight')


def plot_traj_inte(x0, data, ax, integrator='AB', m=1, seed=0, steps = 10):
    h=0.002
    t = np.linspace(0, h*steps, steps+1)
    
    local = '.\\GOoutputs\\GO_'+integrator+'_{}_{}_{}'.format(m, 1, seed)
    net = torch.load(local + '\\model_best.pkl', map_location='cpu')
    flow_pred = net.predict(torch.tensor(x0, dtype=torch.float32), h, steps, keepinitx=True, returnnp=True)
    
    ax.plot(t, flow_pred[0:, 0], color='red', label='LMNets', linestyle='--', dashes=(3,3), linewidth= linewidth, zorder=2)

    if integrator=='AM':
        ax.set_title(integrator+'{}'.format(m-1),fontsize=titlesize,loc='center')
    else:
        ax.set_title(integrator+'{}'.format(m), fontsize=titlesize, loc= 'center')  
    ax.set_xlim(0,10)
    ax.set_yticks([0.5,1,1.5])
    ax.set_xlabel(r'$T$', fontsize=xsize, loc='right')
    ax.set_ylabel(r'$S_1$', fontsize=xsize+3)
    ax.tick_params(labelsize=ticksize)

if __name__=='__main__':

    
    plot_traj()
    plot_error()
