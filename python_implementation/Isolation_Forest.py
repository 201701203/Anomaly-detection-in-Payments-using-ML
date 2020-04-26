"""
     Group 18 Isolation Forest
"""
import numpy as np
import random as rn


def c_factor(n) :
    return 2.0*(np.log(n-1)+0.5772156649) - (2.0*(n-1.)/(n*1.0))


class Node(object):
    def __init__(self, X, q, p, e, left, right, node_type = '' ):
        self.e = e
        self.size = len(X)
        self.X = X # to be removed
        self.q = q
        self.p = p
        self.left = left
        self.right = right
        self.ntype = node_type



class iTree(object):
    def __init__(self,X,e,l):
        self.e = e # depth
        self.X = X #save data for now
        self.size = len(X) #  n objects
        self.Q = np.arange(np.shape(X)[1], dtype='int') # n dimensions
        self.l = l # depth limit
        self.p = None
        self.q = None
        self.exnodes = 0
        self.root = self.make_tree(X,e,l)
        

    def make_tree(self,X,e,l):
        self.e = e
        if e >= l or len(X) <= 1:
            left = None
            right = None
            self.exnodes += 1
            return Node(X, self.q, self.p, e, left, right, node_type = 'exNode' )
        else:
            self.q = rn.choice(self.Q)
            mini = X[:,self.q].min()
            maxi = X[:,self.q].max()
            if mini==maxi:
                left = None
                right = None
                self.exnodes += 1
                return Node(X, self.q, self.p, e, left, right, node_type = 'exNode' )
            self.p = rn.uniform(mini,maxi)
            w = np.where(X[:,self.q] < self.p,True,False)
            return Node(X, self.q, self.p, e,\
            left=self.make_tree(X[w],e+1,l),\
            right=self.make_tree(X[~w],e+1,l),\
            node_type = 'inNode' )



class PathFactor(object):
    def __init__(self,x,itree):
        self.path_list=[]        
        self.x = x
        self.e = 0
        self.path = self.find_path(itree.root)

    def find_path(self,T):
        if T.ntype == 'exNode':
            if T.size == 1: return self.e
            else:
                self.e = self.e + c_factor(T.size)
                return self.e
        else:
            a = T.q
            self.e += 1
            if self.x[a] < T.p:
                self.path_list.append('L')
                return self.find_path(T.left)
            else:
                self.path_list.append('R')
                return self.find_path(T.right)



class iForest(object):
    def __init__(self,X, ntrees,  sample_size, limit=None):     # constructor
        self.ntrees = ntrees
        self.X = X
        self.nobjs = len(X)
        self.sample = sample_size
        self.Trees = []
        self.limit = limit
        if limit is None:
            self.limit = int(np.ceil(np.log2(self.sample)))
        self.c = c_factor(self.sample)        
        for _ in range(self.ntrees):
            ix = rn.sample(range(self.nobjs), self.sample)      # give random numbers from 1 to n
            X_p = X.values[ix,:]
            self.Trees.append(iTree(X_p, 0, self.limit))

    def predict(self, X_in = None, thresold = 0.7):
        if X_in is None:
            X_in = self.X
        S = np.zeros(len(X_in))
        y_pre = np.zeros(len(X_in))
        for i in range(len(X_in)):
            h_temp = 0
            for j in range(self.ntrees):
                h_temp += PathFactor(self.X.values[i,:],self.Trees[j]).path*1.0
            Eh = h_temp/self.ntrees
            S[i] = 2.0**(-Eh/self.c)
            if S[i] > thresold :
                y_pre[i] = 1
            else :
                y_pre[i] = 0
        return y_pre

