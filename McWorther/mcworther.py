import numpy as np
import matplotlib.pyplot as plt


class BrooksCorey:
    def __init__(self, pd, lambda_):
        self.pd = pd
        self.lambda_ = lambda_

    def pc(self, Sw): 
        return self.pd*Sw ** (-1/self.lambda_)

    def dPcdSw(self, Sw): 
        return -self.pd/self.lambda_*Sw ** (-1/self.lambda_-1)

    # Relative permeabilities
    def kn(self, Sw): 
        return  (1-Sw) ** 2 * (1-Sw ** ((2+self.lambda_)/self.lambda_));

    def kw(self, Sw): 
        return  Sw ** ((2+3*self.lambda_)/self.lambda_);

    def Sw(self, pc):
        return (self.pd/pc) ** self.lambda_

    def plot(self):
        plot_model(self.pc, self.kn, self.kw)


class VanGenuchten:
    def __init__(self, alpha=0.0001, n=10):
        self.alpha = alpha
        self.n = n
        self.m = 1 - 1 / n

    def pc(self, Sw):
        return (1 / self.alpha) * (Sw ** (-1 / self.m) - 1) ** (1 / self.n)

    def dPcdSw(self, Sw):
        return -1 / (self.alpha * Sw * self.n * self.m) * Sw ** (-1 / self.m) * (Sw ** (-1 / self.m) - 1) ** (1 / self.n - 1)

    # Relative permeabilities
    def kn(self, Sw):
        return (1 - Sw) ** (1 / 3) * (1 - Sw ** (1 / self.m)) ** (2 * self.m)

    def kw(self, Sw):
        return np.sqrt(Sw) * (1 - (1 - Sw ** (1 / self.m)) ** self.m) ** 2

    def plot(self):
        plot_model(self.pc, self.kn, self.kw)


def plot_model(pc, kn, kw, Smin=0, Smax=1, nel=100):

        eps2 = 10 ** -16
        Sw = np.linspace(Smin+eps2, Smax-eps2, nel)

        plt.subplot(1, 2, 1)
        plt.semilogy(Sw, pc(Sw))
        plt.xlabel("Sw")
        plt.ylabel("Pc")

        plt.subplot(1, 2, 2)
        plt.plot(Sw, kn(Sw))
        plt.plot(Sw, kw(Sw))
        plt.xlabel("Sw")
        plt.ylabel("kn, kw")

        plt.show()
    
class McWorther:
    def __init__(self, model, phi=0.5, K=1.0e-10, muw=0.001, mun=0.005, S0=0.9, Si=0.0, t=1000, nel=1000, max_iter=100000, eps=1.0e-14):
          
        eps2 = 10 ** -16

        # Function f and D (McWhorter and Sunada)
        def f(Sw):
            return 1 / (1 + (model.kn(Sw) * muw) / (model.kw(Sw) * mun))
        
        def D(Sw):
            return -(K * model.kn(Sw) * f(Sw)) / mun * model.dPcdSw(Sw)       

        self.f = f
        self.D = D

        def integral(x):
            return np.sum(x)

        Sw = np.linspace(Si+eps, S0-eps, nel)
        dSw = Sw[1] - Sw[0]

        # Iterative computation of F
        F = np.ones(nel)
        F[0] = eps2
        Falt = F.copy()

        Di = D(Sw)

        for it in range(max_iter):
            a = integral((Sw[1:] - Si) * Di[1:] / F[1:])

            for i in range(1, len(Sw) - 1):
                b = integral((Sw[i:] - Sw[i]) * Di[i:] / F[i:])
                F[i] = 1 - b / a
            
            if np.linalg.norm(F - Falt) < eps:
                break
            Falt = F.copy()
            self.A = np.sqrt(phi / 2 * a * dSw)
            
        dFdS = np.zeros_like(F)
        for i in range(len(Sw)):
            dFdS[i] = integral(Di[i:] / F[i:]) / a

        x = 2 * self.A / phi * dFdS * np.sqrt(t)  
        
        self.x = x[1:]
        self.Sw = Sw[1:]

    def plot_solution(self):

        plt.plot(self.x, self.Sw, label=f"Analytical solution A={self.A:.4e}")
        plt.xlabel("x")
        plt.ylabel("Sw")
        plt.legend()
        #plt.show()


    def plot_D_f(self, nel=100):

        eps2 = 10 ** -16
        Sw = np.linspace(0+eps2, 1-eps2, nel)

        plt.subplot(1, 2, 1)
        plt.plot(Sw, self.D(Sw))
        plt.xlabel('Sw')
        plt.ylabel('D')

        plt.subplot(1, 2, 2)
        plt.plot(Sw, self.f(Sw))
        plt.xlabel('Sw')
        plt.ylabel('f')

        plt.show()



     

