import numpy as np
class gaussian:

    def logp(self, q):

        x=q
        lp = -np.log(2*np.pi)-(x**2)/2
        return lp

    def dlogp(self, q):
        x=q
        dlogf =  -np.array([x ])
        return dlogf

    def d2logp(self, x):
        d2f = -np.array([ [ 1] ])
        return d2f

    def densities(self):
        density = lambda x: stats.norm.pdf(x )
        return density, density

    def hvp_logp(self, x, v):
        return self.dlogp(v)