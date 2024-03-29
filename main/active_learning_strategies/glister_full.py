from .strategy import Strategy
import copy
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader

import math
import random

class GLISTER(Strategy):
    """
    Implementation of GLISTER-ACTIVE Strategy :footcite:`killamsetty2020glister`. 
    This class extends :class:`active_learning_strategies.strategy.Strategy`.
    
    Parameters
    ----------
    X: Numpy array 
        Features of the labled set of points 
    Y: Numpy array
        Lables of the labled set of points 
    unlabeled_x: Numpy array
        Features of the unlabled set of points 
    net: class object
        Model architecture used for training. Could be instance of models defined in `distil.utils.models` or something similar.
    handler: class object
        It should be a subclass of torch.utils.data.Dataset i.e, have __getitem__ and __len__ methods implemented, so that is could be passed to pytorch DataLoader.Could be instance of handlers defined in `distil.utils.DataHandler` or something similar.
    nclasses: int 
        No. of classes in tha dataset
    args: dictionary
        This dictionary should have keys 'batch_size' and  'lr'. 
        'lr' should be the learning rate used for training. 'batch_size'  'batch_size' should be such 
        that one can exploit the benefits of tensorization while honouring the resourse constraits.
    valid: boolean
        Whether validation set is passed or not
    X_val: Numpy array, optional
        Features of the points in the validation set. Mandatory if `valid=True`.
    Y_val:Numpy array, optional
        Lables of the points in the validation set. Mandatory if `valid=True`.
    loss_criterion: class object, optional
        The type of loss criterion. Default is **torch.nn.CrossEntropyLoss()**
    typeOf: str, optional
        Determines the type of regulariser to be used. Default is **'none'**.
        For random regulariser use **'Rand'**.
        To use Facility Location set functiom as a regulariser use **'FacLoc'**.
        To use Diversity set functiom as a regulariser use **'Diversity'**.
    lam: float, optional
        Determines the amount of regularisation to be applied. Mandatory if is not `typeOf='none'` and by default set to `None`.
        For random regulariser use values should be between 0 and 1 as it determines fraction of points replaced by random points.
        For both 'Diversity' and 'FacLoc', `lam` determines the weightage given to them while computing the gain.
    kernel_batch_size: int, optional
        For 'Diversity' and 'FacLoc' regualrizer versions, similarity kernel is to be computed, which 
        entails creating a 3d torch tensor of dimenssions kernel_batch_size*kernel_batch_size*
        feature dimenssion.Again kernel_batch_size should be such that one can exploit the benefits of 
        tensorization while honouring the resourse constraits. 
    """
    

    def __init__(self,X, Y,unlabeled_x, net, handler, nclasses, args,valid,X_val=None,Y_val=None,\
        loss_criterion=nn.CrossEntropyLoss(),typeOf='none',lam=None,kernel_batch_size = 200): # 
        super(GLISTER, self).__init__(X, Y, unlabeled_x, net, handler,nclasses, args)

        if valid:
            self.X_Val = X_val
            self.Y_Val = Y_val
        self.loss = loss_criterion
        self.valid = valid
        self.typeOf = typeOf
        self.lam = lam
        self.kernel_batch_size = kernel_batch_size
        #self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def distance(self,x, y, exp = 2):

      n = x.size(0)
      m = y.size(0)
      d = x.size(1)

      x = x.unsqueeze(1).expand(n, m, d)
      y = y.unsqueeze(0).expand(n, m, d)

      if self.typeOf == "FacLoc":
          dist = torch.pow(x - y, exp).sum(2) 
      elif self.typeOf == "Diversity":
          dist = torch.exp((-1 * torch.pow(x - y, exp).sum(2))/2)
      
      return dist 

    def _compute_similarity_kernel(self):
        
        g_is = []
        for item in range(math.ceil(len(self.grads_per_elem) / self.kernel_batch_size)):
            inputs = self.grads_per_elem[item *self.kernel_batch_size:(item + 1) *self.kernel_batch_size]
            g_is.append(inputs)

        with torch.no_grad():
            
            new_N = len(self.grads_per_elem)
            self.sim_mat = torch.zeros([new_N, new_N], dtype=torch.float32).to(self.device)
            first_i = True
            for i, g_i in enumerate(g_is, 0):
                if first_i:
                    size_b = g_i.size(0)
                    first_i = False
                for j, g_j in enumerate(g_is, 0):
                    self.sim_mat[i * size_b: i * size_b + g_i.size(0),
                    j * size_b: j * size_b + g_j.size(0)] = self.distance(g_i, g_j)

            if self.typeOf == "FacLoc":
                const = torch.max(self.sim_mat).item()
                #self.sim_mat = const - self.sim_mat

                self.min_dist = (torch.ones(new_N, dtype=torch.float32)*const).to(self.device)

    def _compute_per_element_grads(self):
        
        self.grads_per_elem = self.get_grad_embedding(self.unlabeled_x)
        self.prev_grads_sum = torch.sum(self.get_grad_embedding(self.X,self.Y),dim=0).view(1, -1)

    def _update_grads_val(self,grads_currX=None, first_init=False):

        embDim = self.model.get_embedding_dim()
        
        if first_init:
            if self.valid:
                if self.X_Val is not None:
                    loader = DataLoader(self.handler(self.X_Val,self.Y_Val,select=False),shuffle=False,\
                        batch_size=self.args['batch_size'])
                    self.out = torch.zeros(self.Y_Val.shape[0], self.target_classes).to(self.device)
                    self.emb = torch.zeros(self.Y_Val.shape[0], embDim).to(self.device)
                else:
                    raise ValueError("Since Valid is set True, please pass a appropriate Validation set")
            
            else:
                predicted_y = self.predict(self.unlabeled_x)
                self.X_new = np.concatenate((self.unlabeled_x,self.X), axis = 0)
                self.Y_new = np.concatenate((predicted_y,self.Y), axis = 0)

                loader = DataLoader(self.handler(self.X_new,self.Y_new,select=False),shuffle=False,\
                    batch_size=self.args['batch_size'])
                self.out = torch.zeros(self.Y_new.shape[0], self.target_classes).to(self.device)
                self.emb = torch.zeros(self.Y_new.shape[0], embDim).to(self.device)

            self.grads_val_curr = torch.zeros(self.target_classes*(1+embDim), 1).to(self.device)
            
            with torch.no_grad():

                for x, y, idxs in loader:
                    x = x.to(self.device)
                    y = y.to(self.device)
                    init_out, init_l1 = self.model(x,last=True)
                    self.emb[idxs] = init_l1 
                    for j in range(self.target_classes):
                        try:
                            self.out[idxs, j] = init_out[:, j] - (1 * self.args['lr'] * (torch.matmul(init_l1, self.prev_grads_sum[0][(j * embDim) +
                                    self.target_classes:((j + 1) * embDim) + self.target_classes].view(-1, 1)) + self.prev_grads_sum[0][j])).view(-1)
                        except KeyError:
                            print("Please pass learning rate used during the training")
                
                    scores = F.softmax(self.out[idxs], dim=1)
                    one_hot_label = torch.zeros(len(y), self.target_classes).to(self.device)
                    one_hot_label.scatter_(1, y.view(-1, 1), 1)
                    l0_grads = scores - one_hot_label
                    l0_expand = torch.repeat_interleave(l0_grads, embDim, dim=1)
                    l1_grads = l0_expand * init_l1.repeat(1, self.target_classes)

                    self.grads_val_curr += torch.cat((l0_grads, l1_grads), dim=1).sum(dim=0).view(-1, 1)
            
            if self.valid:
                self.grads_val_curr /= self.Y_Val.shape[0]
            else:
                self.grads_val_curr /= predicted_y.shape[0]

        elif grads_currX is not None:
            # update params:
            with torch.no_grad():

                for j in range(self.target_classes):
                    try:
                        self.out[:, j] = self.out[:, j] - (1 * self.args['lr'] * (torch.matmul(self.emb, grads_currX[0][(j * embDim) +
                                    self.target_classes:((j + 1) * embDim) + self.target_classes].view(-1, 1)) +  grads_currX[0][j])).view(-1)
                    except KeyError:
                        print("Please pass learning rate used during the training")

            
                scores = F.softmax(self.out, dim=1)
                if self.valid:
                    Y_Val = torch.tensor(self.Y_Val,device=self.device)
                    one_hot_label = torch.zeros(Y_Val.shape[0], self.target_classes).to(self.device)
                    one_hot_label.scatter_(1,Y_Val.view(-1, 1), 1)   
                else:
                    
                    one_hot_label = torch.zeros(self.Y_new.shape[0], self.target_classes).to(self.device)
                    one_hot_label.scatter_(1, torch.tensor(self.Y_new,device=self.device).view(-1, 1), 1)
                l0_grads = scores - one_hot_label
                l0_expand = torch.repeat_interleave(l0_grads, embDim, dim=1)
                l1_grads = l0_expand * self.emb.repeat(1, self.target_classes)

                self.grads_val_curr = torch.cat((l0_grads, l1_grads), dim=1).mean(dim=0).view(-1, 1)

    def eval_taylor_modular(self, grads,greedySet=None,remset=None):

        with torch.no_grad():
            if self.typeOf == "FacLoc":
                gains = torch.matmul(grads, self.grads_val_curr) + self.lam*((self.min_dist - \
                    torch.min(self.min_dist,self.sim_mat[remset])).sum(1)).view(-1, 1).to(self.device)
                
            elif self.typeOf == "Diversity" and len(greedySet) > 0:
                gains = torch.matmul(grads, self.grads_val_curr) - \
                    self.lam*self.sim_mat[remset][:, greedySet].sum(1).view(-1, 1).to(self.device)
            else:
                #Something like this
                ##self.out[:, j] = self.out[:, j] - (1 * self.args['lr'] * (torch.matmul(self.emb, grads_currX[0][(j * embDim) +
                ##self.target_classes:((j + 1) * embDim) + self.target_classes].view(-1, 1)) +  grads_currX[0][j])).view(-1)
                gains = torch.matmul(grads, self.grads_val_curr)
        return gains
    
    def select(self, budget):

        """
        Select next set of points
        
        Parameters
        ----------
        budget: int
            Number of indexes to be returned for next set
        
        Returns
        ----------
        chosen: list
            List of selected data point indexes with respect to unlabeled_x
        """ 

        self._compute_per_element_grads()
        self._update_grads_val(first_init=True)
        
        numSelected = 0
        greedySet = list()
        remainSet = list(range(self.unlabeled_x.shape[0]))

        if self.typeOf == 'Rand':
            if self.lam is not None:
                if self.lam >0 and self.lam < 1:
                    curr_bud = (1-self.lam)*budget
                else:
                    raise ValueError("Lambda value should be between 0 and 1")
            else:
                raise ValueError("Please pass a appropriate lambda value for random regularisation")
        else:
            curr_bud = budget

        if self.typeOf == "FacLoc" or self.typeOf == "Diversity":
            if self.lam is not None:
                self._compute_similarity_kernel()
            else:
                if self.typeOf == "FacLoc":
                    raise ValueError("Please pass a appropriate lambda value for Facility Location based regularisation")
                elif self.typeOf == "Diversity":
                    raise ValueError("Please pass a appropriate lambda value for Diversity based regularisation")
        
        while (numSelected < curr_bud):

            if self.typeOf == "Diversity":
                gains = self.eval_taylor_modular(self.grads_per_elem[remainSet],greedySet,remainSet)
            elif self.typeOf == "FacLoc":
                gains = self.eval_taylor_modular(self.grads_per_elem[remainSet],remset=remainSet)
            else:
                gains = self.eval_taylor_modular(self.grads_per_elem[remainSet])#rem_grads)
                
            bestId = remainSet[torch.argmax(gains).item()]
            greedySet.append(bestId)
            remainSet.remove(bestId)
            numSelected += 1
            
            self._update_grads_val(self.grads_per_elem[bestId].view(1, -1))

            if self.typeOf == "FacLoc":
                self.min_dist = torch.min(self.min_dist,self.sim_mat[bestId])
            
        if self.typeOf == 'Rand':
            greedySet.extend(list(np.random.choice(remainSet, size=budget - int(curr_bud),replace=False)))
        
        return greedySet