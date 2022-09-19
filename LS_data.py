import numpy as np

import learner as ln
from learner.integrator.rungekutta import RK4
from utils import reshape

class LSIMDE:
    def __init__(self, integrator='AB', m=2, h=0.008):
        self.h=h
        self.solver = RK4(self.F, N=10)   
        coe={  'AB': {'1': [1/2, 1/6, 1/6,  1/24,   1/8,   1/24,   1/24],        
                      '2': [0, 5/12, 5/12, -1/4, -3/4, -1/4, -3/4],
                      '3': [0, 0, 0, 3/8, 9/8, 3/8, 3/8,
                            -289/720, -289/180, -289/360, -289/360, -289/720,
                            -289/720, -289/240, -289/720, -289/720]},
                'AM': {'1': [-1/2, 1/6, 1/6, -1/24, -1/8, -1/24, -1/24],
                      '2': [0, -1/12, -1/12, 0, 0, 0, 0],
                      '3': [0, 0, 0, -1/24, -1/8, -1/24, -1/24, 
                            11/720, 11/180, 11/360, 11/360, 11/720, 11/720,
                            11/240, 11/720, 11/720]},
                'Ny': {'1': [0, 1/3, 1/3, -1/3, -1, -1/3, -1/3],
                      '2': [0, 0, 0, 1/3, 1, 1/3, 1/3]},
                'MS': {'1': [-2, 16/3, 16/3, -46/3, -46, -46/3, -46/3],
                      '2': [0, 1/3, 1/3, -1/3, -1, -1/3, -1/3]},
                'BDF': {'1': [-1/2, 1/6, 1/6, -1/24, -1/8, -1/24, -1/24],
                        '2': [0, -1/3, -1/3, 1/4, 3/4, 1/4, 1/4],
                        '3': [0, 0, 0, -1/4, -3/4, -1/4, -1/4,
                              3/10, 6/5, 3/5, 3/5, 3/10, 3/10, 9/10, 3/10, 3/10]}
                }
        self.a = coe[integrator][str(m)]
     
    def F(self, y):
        h=self.h
        y=y*10
        f=np.zeros(y.shape)
        f[...,0]=10*y[...,1]-10*y[...,0]
        f[...,1]=28*y[...,0]-y[...,0]*y[...,2]-y[...,1]
        f[...,2]=y[...,0]*y[...,1]-8/3*y[...,2]
        
        dff=np.zeros(y.shape)
        dff[...,0] = -10*f[...,0] + 10* f[...,1]
        dff[...,1] = (28- y[...,2])* f[...,0]-f[...,1]-y[...,0]*f[...,2]
        dff[...,2] = y[...,1]* f[...,0] + y[...,0]*f[...,1]-8/3*f[...,2]
        
        dfdff=np.zeros(y.shape)
        dfdff[...,0] = -10*dff[...,0] + 10* dff[...,1]
        dfdff[...,1] = (28- y[...,2])* dff[...,0]-dff[...,1]- y[...,0]*dff[...,2]
        dfdff[...,2] =  y[...,1]* dff[...,0]+ y[...,0]*dff[...,1]-8/3*dff[...,2]
        
        ddfff=np.zeros(y.shape)
        ddfff[...,0]=0
        ddfff[...,1]=-2*f[...,0]*f[...,2]
        ddfff[...,2]=2*f[...,0]*f[...,1]
#       

        dddffff = np.zeros(y.shape) 
        
        ddfdfff=np.zeros(y.shape)
        ddfdfff[...,0]=0
        ddfdfff[...,1]=-dff[...,0]*f[...,2]-dff[...,2]*f[...,0]
        ddfdfff[...,2]=dff[...,0]*f[...,1]+dff[...,1]*f[...,0]
        
        dfddfff=np.zeros(y.shape)
        dfddfff[...,0] = -10*ddfff[...,0] + 10* ddfff[...,1]
        dfddfff[...,1] = (28- y[...,2])* ddfff[...,0]-ddfff[...,1]- y[...,0]*ddfff[...,2]
        dfddfff[...,2] =  y[...,1]* ddfff[...,0]+ y[...,0]*f[...,1]-8/3*ddfff[...,2]
        
        dfdfdff=np.zeros(y.shape)
        dfdfdff[...,0] = -10*dfdff[...,0] + 10* dfdff[...,1]
        dfdfdff[...,1] = (28- y[...,2])* dfdff[...,0]-dfdff[...,1]- y[...,0]*dfdff[...,2]
        dfdfdff[...,2] =  y[...,1]* dfdff[...,0]+ y[...,0]*dfdff[...,1]-8/3*dfdff[...,2]
        
        return (
                f #f
                +self.a[0]* h    * dff     #f'f
                +self.a[1]* h**2 * ddfff   #f''(f,f)
                +self.a[2]* h**2 * dfdff   #f'f'f
                +self.a[3]* h**3 * dddffff #f'''(f,f,f)
                +self.a[4]* h**3 * ddfdfff #f''(f'f,f)
                +self.a[5]* h**3 * dfddfff #f'f''(f,f)
                +self.a[6]* h**3 * dfdfdff #f'f'f'f 
                )/10






class LSData(ln.Data):
    def __init__(self, T=10, substeps=4, x0=[-0.8,0.7,2.7], index=2):
        '''
        h=0.002*index
        length=T/h

        '''
        super(LSData, self).__init__()
        self.solver = RK4(self.f, N=10)
        self.x0 = x0
        self.index = int(index)
        self.T=T
        self.substeps = substeps
        self.__init_data()
    
    def f(self, y):
        y=y*10
        f = np.ones_like(y)
        sig ,r, b = 10, 28, 8/3
        f[...,0] = sig * (y[...,1]-y[...,0])
        f[...,1]= -y[...,0]*y[...,2]+r *y[...,0] - y[...,1]
        f[...,2] = y[...,0]*y[...,1] - b*y[...,2]
        return f/10
    
    @property
    def dim(self):
        return 3
     
    
    def __init_data(self):        
        X = self.solver.flow(np.array(self.x0), 0.002, int(self.T/0.002))
        X_train = X[::self.index]
        self.X_train = reshape(X_train, substeps = self.substeps)
        
        if not self.index==1:
            Y=[]
            for _ in range(0, self.index):
                traj = X[_::self.index]
                Y_ = reshape(traj, substeps = self.substeps)
                Y.append(Y_)
            self.X_test = np.hstack(Y)
        else:
            self.X_test = self.X_train
