import os
import time
import numpy as np
import torch

from .nn import LossNN
from .utils import timing, cross_entropy_loss

class Brain:
    '''Runner based on torch.
    '''
    brain = None
    
    @classmethod
    def Init(cls, filename, data, net, criterion, optimizer, lr, iterations, lr_decay=1, batch_size=None, 
             print_every=1000, save=False, callback=None, dtype='float', device='cpu'):
        cls.brain = cls(filename, data, net, criterion, optimizer, lr, lr_decay, iterations, batch_size, 
                         print_every, save, callback, dtype, device)
        
    @classmethod
    def Run(cls):
        cls.brain.run()
        
    @classmethod
    def Restore(cls):
        cls.brain.restore()
        
    @classmethod
    def Output(cls, data=True, best_model=True, loss_history=True, info=None, path=None, **kwargs):
        cls.brain.output(data, best_model, loss_history, info, path, **kwargs)
    
    @classmethod
    def Loss_history(cls):
        return cls.brain.loss_history
    
    @classmethod
    def Encounter_nan(cls):
        return cls.brain.encounter_nan
    
    @classmethod
    def Best_model(cls):
        return cls.brain.best_model
    
    def __init__(self, filename, data, net, criterion, optimizer, lr, lr_decay, iterations, batch_size, 
                 print_every, save, callback, dtype, device):
        self.filename=filename
        self.data = data
        self.net = net
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr = lr
        self.lr_decay=lr_decay
        self.iterations = iterations
        self.batch_size = batch_size
        self.print_every = print_every
        self.save = save
        self.callback = callback
        self.dtype = dtype
        self.device = device
        
        self.loss_history = None
        self.encounter_nan = False
        self.best_model = None
        
        self.__optimizer = None
        self.__criterion = None
    
    @timing
    def run(self):
        self.__init_brain()
        print('Training...', flush=True)
        loss_history = []
        if not os.path.isdir('./'+'training file/'+self.filename+'/model'): os.makedirs('./'+'training file/'+self.filename+'/model')

        for i in range(self.iterations + 1):
            if self.batch_size is not None:
                mask = np.random.choice(self.data.X_train.size(1), self.batch_size, replace=False)
                loss = self.__criterion(self.net(self.data.X_train[:, mask]), None)
            else:
                loss = self.__criterion(self.net(self.data.X_train), self.data.y_train)
            if i % self.print_every == 0 or i == self.iterations:
                loss_test = self.__criterion(self.net(self.data.X_test), self.data.y_test)
                loss_history.append([i, loss.item(), loss_test.item()])
                print('{:<9}Train loss: {:<25}Test loss: {:<25}'.format(i, loss.item(), loss_test.item()), flush=True)
                print('lr',self.__optimizer.param_groups[0]['lr'])
                if torch.any(torch.isnan(loss)):
                    self.encounter_nan = True
                    print('Encountering nan, stop training', flush=True)
                    return None                
                if self.save:                   
                    torch.save(self.net, 'training file/'+self.filename+'/model/model{}.pkl'.format(i))
                if self.callback is not None: 
                    to_stop = self.callback(self.data, self.net)
                    if to_stop: break
            if i < self.iterations:
                self.__optimizer.zero_grad()
                loss.backward()
                self.__optimizer.step()
            if self.lr_decay!=1:
                self.__optimizer.param_groups[0]['lr']=self.__optimizer.param_groups[0]['lr']/self.lr_decay**(1/self.iterations)
            loss_record = np.array(loss_history)
            np.savetxt('training file/'+self.filename+'/loss.txt', loss_record)
        self.loss_history = np.array(loss_history)
        print('Done!', flush=True)
        return self.loss_history
    
    def restore(self):
        if self.loss_history is not None and self.save == True:
            best_loss_index = np.argmin(self.loss_history[:, 1])
            iteration = int(self.loss_history[best_loss_index, 0])
            loss_train = self.loss_history[best_loss_index, 1]
            loss_test = self.loss_history[best_loss_index, 2]
            print('Best model at iteration {}:'.format(iteration), flush=True)
            print('Train loss:', loss_train, 'Test loss:', loss_test, flush=True)
            
            path = './outputs/' + self.filename
            if not os.path.isdir('./outputs/'+self.filename): os.makedirs('./outputs/'+self.filename)
            f = open(path +'/output.txt',mode='a')
            f.write('\n\n'
                    +'Train completion time  '
                    + time.strftime('%Y-%m-%d-%H-%M-%S',time.localtime(time.time()))
                    + '\n'
                    + 'Best model at iteration: {}'.format(iteration)
                    + '\n'
                    + 'Train loss: %s'%(loss_train)
                    + '\n'
                    + 'Test loss: %s'%(loss_test)
                    )
            f.close()
            
            self.best_model = torch.load('training file/'+self.filename+'/model/model{}.pkl'.format(iteration))
        else:
            raise RuntimeError('restore before running or without saved models')
        return self.best_model
    
    def output(self, data, best_model, loss_history, info, path, **kwargs):
        if path is None:
            path = './outputs/' + self.filename
        if not os.path.isdir(path): os.makedirs(path)
        if best_model:
            torch.save(self.best_model, path + '/model_best.pkl')
        if loss_history:
            np.savetxt(path + '/loss.txt', self.loss_history)
        if info is not None:
            with open(path + '/info.txt', 'w') as f:
                for item in info:
                    f.write('{}: {}\n'.format(item[0], str(item[1])))
        for key, arg in kwargs.items():
            np.savetxt(path + '/' + key + '.txt', arg)
        
            
    def __init_brain(self):
        self.loss_history = None
        self.encounter_nan = False
        self.best_model = None
        self.data.device = self.device
        self.data.dtype = self.dtype
        self.net.device = self.device
        self.net.dtype = self.dtype
        self.net.InitNet()
        self.__init_optimizer()
        self.__init_criterion()

    def __init_optimizer(self):
        if self.optimizer == 'adam':
            self.__optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        else:
            raise NotImplementedError
            
    def __init_criterion(self):
        if isinstance(self.net, LossNN):
            self.__criterion = self.net.criterion
            if self.criterion is not None:
                raise Warning('loss-oriented neural network has already implemented its loss function')
        elif self.criterion == 'MSE':
            self.__criterion = torch.nn.MSELoss()
        elif self.criterion == 'CrossEntropy':
            self.__criterion = cross_entropy_loss
        else:
            raise NotImplementedError   