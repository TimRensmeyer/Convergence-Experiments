class Metric():
    def __init__(self, model,lamb,alpha):
        self.ema=[torch.ones(size=p.shape) for p in model.parameters()]
        self.alpha=alpha
        self.lamb=lamb
    def forward(self,model):
        G=[]
        Gamma=[]

        for i,(p,v) in enumerate(zip(model.parameters(),self.ema)):
            v=self.alpha*v+(1-self.alpha)*(p.grad**2)
            g=1/((v+self.lamb**2)**0.5)                     
            gam=torch.autograd.grad(g,p)[0].detach()        
            v=v.detach()
            self.ema[i]=v
            G.append(g)
            Gamma.append(gam/(1-self.alpha))
        return G,Gamma
    

class Fixed_PSGLD():
    def __init__(self, model,a,lamb,alpha,stepsize):
        self.a=a
        self.lamb=lamb
        self.stepsize=stepsize
        self.G=Metric(model,lamb,alpha)

    def step(self,model):
        G,Gamma=self.G.forward(model)
        for g,p,gam in zip(G,model.parameters(),Gamma):
            noise=g**0.5*torch.normal(0,torch.ones(size=p.shape)*self.stepsize**0.5)
            dp=0.5*self.stepsize*(g*p.grad+gam) + noise
            new_p=p+dp
            if dp>1:
                print(p,g,g*p.grad,gam,noise)
            p.requires_grad=False
            p.copy_(new_p.detach())
            p.requires_grad=True

        model.zero_grad()
