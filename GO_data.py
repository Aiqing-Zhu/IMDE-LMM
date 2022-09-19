import numpy as np
import os

import learner as ln
from learner.integrator.rungekutta import RK4
from utils import reshape

class GOData(ln.Data):
    '''Data for learning glycolyti oscillator
    '''
    def __init__(self, train_traj_num=1, test_traj_num=1, substeps=2, T=1, index=1, 
                 seed=0,
                 region = [[0.15, 1.60], [0.19,2.16,], [0.04,0.20], [0.10,0.35], [0.08,0.30], [0.14,2.67], [0.05,0.10]],
                 gene_traj=False
                 ):
        
        super(GOData, self).__init__()
        self.train_traj_num = train_traj_num
        self.test_traj_num = test_traj_num
        self.substeps =substeps
        self.solver = RK4(self.f, N=10)
        self.T=T
        self.index=int(index)
        self.seed=seed
        for a in region:
            diff = a[1]-a[0]
            a[0]=a[0]+diff/4
            a[1]=a[1]-diff/4
        self.region = region
        
        if gene_traj==True:
            self.__init_traj()
            
        self.__init_data()
       
    
    @property
    def dim(self):
        return 7

    def f(self, x):
        J0 = 2.5
        k1 = 100.0
        k2 = 6.0
        k3 = 16.0
        k4 = 100.0
        k5 = 1.28
        k6 = 12.0
        k = 1.8
        kappa = 13.0
        q = 4
        K1 = 0.52
        psi = 0.1
        N = 1.0
        A = 4.0
        
        f = np.ones(x.shape, dtype=np.float32)
        f[...,0] = J0 - (k1*x[...,0]*x[...,5])/(1 + (x[...,5]/K1)**q)
        f[...,1] = 2*(k1*x[...,0]*x[...,5])/(1 + (x[...,5]/K1)**q) - k2*x[...,1]*(N-x[...,4]) - k6*x[...,1]*x[...,4]
        f[...,2] = k2*x[...,1]*(N-x[...,4]) - k3*x[...,2]*(A-x[...,5])
        f[...,3] = k3*x[...,2]*(A-x[...,5]) - k4*x[...,3]*x[...,4] - kappa*(x[...,3]-x[...,6])
        f[...,4] = k2*x[...,1]*(N-x[...,4]) - k4*x[...,3]*x[...,4] - k6*x[...,1]*x[...,4]
        f[...,5] = -2*(k1*x[...,0]*x[...,5])/(1 + (x[...,5]/K1)**q) + 2*k3*x[...,2]*(A-x[...,5]) - k5*x[...,5]
        f[...,6] = psi*kappa*(x[...,3]-x[...,6]) - k*x[...,6]

        return f

    
    def __init_traj(self):
        self.__generate_train_traj(self.train_traj_num)
        self.__generate_test_traj(self.test_traj_num)
        
    def __init_data(self):
        self.X_train = self.__generate_train_data(self.train_traj_num, self.seed)
        self.X_test = self.__generate_test_data(self.test_traj_num, self.seed)
    
    def __generate_train_traj(self, N =2, seed=0):
        region = self.region
        
        X_initial = np.zeros([N,self.dim])
        for i in range(self.dim):
            X_initial[:,i]= np.random.uniform(region[i][0], region[i][1], N)        
        
        for i in range(N):
            if os.path.exists('GOdata/{}GO_train{}.npy'.format(seed, i)):
                print('Traing data has been generated')
            else:
                if not os.path.isdir('./GOdata'): os.makedirs('./GOdata')
                traj = self.solver.flow(X_initial[i], 0.002, int(self.T/0.002))
                np.save('GOdata/{}GO_train{}.npy'.format(seed, i), traj)
    
    def __generate_train_data(self, N =2, seed=0): 

        X_train=[]

        for i in range(N):
            if os.path.exists('GOdata/{}GO_train{}.npy'.format(seed, i)):
                traj = np.load('GOdata/{}GO_train{}.npy'.format(seed, i))
            else:
                print(i)
                raise ValueError('Training data has not been generated')
            xi= reshape(traj[::self.index], self.substeps)
            X_train.append(xi)
        return np.hstack(X_train)



    def __generate_test_traj(self, N =2, seed=0):
        region = self.region
        
        X_initial = np.zeros([N,self.dim])
        for i in range(self.dim):
            X_initial[:,i]= np.random.uniform(region[i][0], region[i][1], N)        
        for i in range(N):
            if os.path.exists('GOdata/{}GO_test{}.npy'.format(seed, i)):
                print('Test data has been generated')
            else:
                if not os.path.isdir('./GOdata'): os.makedirs('./GOdata')
                traj = self.solver.flow(X_initial[i], 0.002, int(self.T/0.002))
                np.save('GOdata/{}GO_test{}.npy'.format(seed, i), traj)
    

    def __generate_test_data(self, N =2, seed=0): 
        region = self.region
        
        X_initial = np.zeros([N,self.dim])
        for i in range(self.dim):
            X_initial[:,i]= np.random.uniform(region[i][0], region[i][1], N)

        X_test=[]
        
        for i in range(N):
            if os.path.exists('GOdata/{}GO_test{}.npy'.format(seed, i)):
                traj = np.load('GOdata/{}GO_test{}.npy'.format(seed, i))
            else:
                raise ValueError('Training data has not been generated')
            for _ in range(0, self.index):
                Y_ = reshape(traj[_::self.index], substeps = self.substeps)
                X_test.append(Y_)
        

        return np.hstack(X_test)



def main():
    data = GOData(gene_traj=True)

if __name__ == '__main__':
    main()
         

