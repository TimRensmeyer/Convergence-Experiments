import torch
import torch.nn as nn

class Metric():
    def __init__(self, model,lamb,alpha):
        self.ema=[torch.zeros(size=p.shape) for p in model.parameters()]
        self.alpha=alpha
        self.lamb=lamb
    def forward(self,model):
        G=[]
        for i,(p,v) in enumerate(zip(model.parameters(),self.ema)):
            v=self.alpha*v+(1-self.alpha)*(p.grad**2)
            G.append(1/(v**0.5+self.lamb))
            self.ema[i]=v
        return G

class AdamSGLD():
    def __init__(self, model,a,lamb,alpha,beta,stepsize):
        self.a=a
        self.lamb=lamb
        self.stepsize=stepsize
        self.beta=beta
        self.momentum=[torch.zeros(size=p.shape) for p in model.parameters()]
        self.G=Metric(model,lamb,alpha)

    def step(self,model):
        G=self.G.forward(model)
        for i,(g,p,m) in enumerate(zip(G,model.parameters(),self.momentum)):
            m=self.beta*m+(1-self.beta)*p.grad
            self.momentum[i]=m
            new_p=p+0.5*self.stepsize*(p.grad+self.a*m*g) + torch.normal(0,torch.ones(size=p.shape)*self.stepsize**0.5)
            p.requires_grad=False
            p.copy_(new_p.detach())
            p.requires_grad=True
        model.zero_grad()
