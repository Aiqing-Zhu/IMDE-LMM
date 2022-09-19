import torch

from .module import LossNN
from .fnn import FNN
from ..integrator.rungekutta import RK4

    
class LMNets(LossNN):
    '''Linear Multistep Method Neural Networks.
    '''
    def __init__(self, dim=4, layers=2, width=128, activation='tanh', initializer='orthogonal', 
                 integrator='AB', m=2, h=0.1):
        super(LMNets, self).__init__()
        self.dim = dim
        self.layers = layers
        self.width = width       
        self.activation = activation
        self.initializer = initializer
        self.h=h
        alpha={'AB': {'1': [-1, 1],        
                      '2': [0, -1, 1],
                      '3': [0, 0, -1, 1],
                      '4': [0, 0, 0, -1, 1]},
               'AM': {'1': [-1, 1],
                      '2': [-1, 1],
                      '3': [0, -1, 1],
                      '4': [0, 0, -1, 1]},
               'Ny': {'1': [-1, 0, 1],
                      '2': [0, -1, 0, 1],
                      '3': [0, 0, -1, 0, 1]},
               'MS': {'1': [-1, 0, 1],
                      '2': [-1, 0, 1],
                      '3': [-1, 0, 1]},
               'BDF': {'1': [-1, 1],
                       '2': [1/2, -2, 3/2],
                       '3': [-1/3, 3/2, -3, 11/6],
                       '4': [1/4, -4/3, 3, -4, 25/12]}
               }
        
        beta={'AB': {'1': [1, 0], #1/2 h^2 y''              
                     '2': [-1/2, 3/2, 0],#5/12 h^3 y'''
                     '3': [5/12, -16/12, 23/12, 0], #3/8 h^4 y^(4)
                     '4': [-9/24, 37/24, -59/24, 55/24, 0] #251/720 h^5 y^(5)
                     },
              'AM': {'1': [0, 1],# -1/2 h^2 y''  
                     '2': [1/2, 1/2], #-1/12 h^3 y''' 
                     '3': [-1/12, 8/12, 5/12], #-1/24 h^4 y^(4)
                     '4': [1/24, -5/24, 19/24, 9/24] #-19/720 h^5y^(5)
                     },
               'Ny': {'1': [0, 2, 0], #1/3 h^3 y^(3)
                      '2': [1/3, -2/3, 7/3, 0], #1/4 h^4 y^(4)
                      '3': [-1/3, 4/3, -5/3, 8/3, 0] #29/90 h^5 y^(5)
                      },
               'MS': {'1': [0, 0, 2], #-2 h^2 y''
                      '2': [0, 2, 0], #1/3 h^3 y^(3)
                      '3': [1/3, 4/3, 1/3]}, #-1/90 h^5 y^(5)
               'BDF': {'1': [0, 1],
                       '2': [0, 0, 1],
                       '3': [0, 0, 0, 1],
                       '4': [0, 0, 0, 0, 1]}
               }

        self.a = torch.tensor(alpha[integrator][str(m)]).unsqueeze(dim=-1).unsqueeze(dim=-1)
        self.b = torch.tensor(beta[integrator][str(m)]).unsqueeze(dim=-1).unsqueeze(dim=-1)
        
        self.modus = self.__init_modules()
    
    def InitNet(self):
        self.a = self.a.to(dtype=self.Dtype, device=self.Device)
        self.b = self.b.to(dtype=self.Dtype, device=self.Device)
        return 0

        
    def criterion(self, x, y):
        return torch.nn.MSELoss()(torch.sum(self.a*x, dim=0)/self.h, torch.sum(self.b*self.vf(x), dim=0))
    
    def predict(self, x0, h=0.1, steps=1, keepinitx=False, returnnp=False):
        solver = RK4(self.vf, N= 10)
        res = solver.flow(x0, h, steps) if keepinitx else solver.flow(x0, h, steps)[..., 1:, :].squeeze()
        return res.cpu().detach().numpy() if returnnp else res
        

    def vf(self, x):
        return self.modus['f'](x)
        
    def __init_modules(self):
        modules = torch.nn.ModuleDict()
        modules['f'] = FNN(self.dim, self.dim, self.layers, self.width, self.activation, self.initializer)
        return modules 
    
    def integrator_loss(self, x, h):
        
        
        torch.nn.MSELoss()(torch.sum(self.a*x /h, dim=0), torch.sum(self.b*self.vf(x), dim=0))
        
