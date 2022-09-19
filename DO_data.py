import numpy as np

import learner as ln
from learner.integrator.rungekutta import RK4
from utils import reshape

class DOIMDE:
    def __init__(self, integrator='Ny', m=1, h=0.01):
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
        true_A = np.array([[-0.1, -2.0], [2.0, -0.1]])
        
        f = y**3@true_A #f
        dff = y**3@ true_A * y**2@(3*true_A)  #f'f
        
        ddfff = (y**3@true_A)**2 * y@(6*true_A)  #f''(f,f)
        dfdff = y**3@ true_A * y**2@(3*true_A) * y**2@(3*true_A) #f'f'f
        
        dddffff = ((y**3@true_A)**3 @ (6*true_A))  #f'''(f,f,f)
        ddfdfff = (y**3@true_A*(y**3@ true_A * y**2@(3*true_A))) * y@(6*true_A) #f''(f'f,f)
        dfddfff = (((y**3@true_A)**2* y@(6*true_A))*  y**2@(3*true_A) ) #f'f''(f,f)
        dfdfdff = (y**3@ true_A * y**2@(3*true_A)* y**2@(3*true_A)* y**2@(3*true_A))#f'f'f'f

        return (
                f #f
                +self.a[0]* h    * dff     #f'f
                +self.a[1]* h**2 * ddfff   #f''(f,f)
                +self.a[2]* h**2 * dfdff   #f'f'f
                +self.a[3]* h**3 * dddffff #f'''(f,f,f)
                +self.a[4]* h**3 * ddfdfff #f''(f'f,f)
                +self.a[5]* h**3 * dfddfff #f'f''(f,f)
                +self.a[6]* h**3 * dfdfdff #f'f'f'f 

                ) 


class DOData(ln.Data):
    def __init__(self, train_traj_num=1, test_traj_num=2, 
                 h=0.01, substeps=1, length=0, 
                 region = [[-2.2, 2.2], [-2.2, 2.2]]
                 ):
        super(DOData, self).__init__()
        self.solver = RK4(self.f, N=10)
        self.h = h
        self.train_traj_num = train_traj_num
        self.test_traj_num = test_traj_num
        self.length = length
        self.substeps = substeps
        self.region = region
        self.__init_data()
    
    def f(self, x):
        true_A = np.array([[-0.1, -2.0], [2.0, -0.1]])
        return x**3@true_A

    
    @property
    def dim(self):
        return 2
    
    def __init_data(self):
        self.X_train = self.__generate_data(self.train_traj_num)
        self.X_test =  self.__generate_data(self.test_traj_num)

    
    def __generate_data(self, N =2): 
        region = self.region
        
        X_initial = np.zeros([N,self.dim])
        for i in range(self.dim):
            X_initial[:,i]= np.random.uniform(region[i][0], region[i][1], N)
        X1=[]
        for i in range(N):
            traj = self.solver.flow(X_initial[i], self.h, self.length)
            xi= reshape(traj, self.substeps)
            X1.append(xi)
        
        x1=np.hstack(X1)
        return x1


    
    
    
    
    
    
    
    
    
    
    
    
