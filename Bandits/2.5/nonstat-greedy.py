import numpy as np
import matplotlib.pyplot as plt

class NSG:
    def __init__(self, epsilon, alpha, delta, tasks, arms):
        self.epsilon = epsilon
        self.alpha = alpha
        self.beta = 1
        self.delta = delta
        self.n = tasks
        self.k = arms

        self.Q = [([0]*self.k) for i in range(self.n)]
        #self.Q2 = [[([0]*self.k) for i in range(self.n)] for y in range(self.delta)]
        self.VQ = [([0]*self.k) for i in range(self.n)]
        self.CQ = [([0]*self.k) for i in range(self.n)]
        self.VN = [([0]*self.k) for i in range(self.n)]
        self.CN = [([0]*self.k) for i in range(self.n)]
        self.VR = [0]*self.delta
        self.CR = [0]*self.delta

    def RW(self, x):
        for i in range(len(x)):
            x[i]=x[i]+np.random.randint(-1,2)
        return x    

    def V(self):
        for i in range(self.delta):
            for j in range(self.n):
                self.Q[j] = self.RW(self.Q[j])
                #self.Q2[i][j] = self.Q[j]
                tmp = np.random.random()
                if tmp <= self.epsilon: arm = np.random.randint(0, self.k)
                else: arm = np.argmax(self.VQ[j])
                R = self.Q[j][arm]
                self.VN[j][arm] = self.VN[j][arm] + 1
                self.VQ[j][arm] = self.VQ[j][arm] + 1/self.VN[j][arm] * (R - self.VQ[j][arm])
                self.VR[i] = self.VR[i] + R/self.n

            if i % 100 == 0:
                print(i)

    
    def C(self):
        self.Q = [([0]*self.k) for i in range(self.n)]
        for i in range(self.delta):
            for j in range(self.n):
                self.Q[j] = self.RW(self.Q[j])
                tmp = np.random.random()
                if tmp <= self.epsilon: arm = np.random.randint(0, self.k)
                else: arm = np.argmax(self.CQ[j])
                R = self.Q[j][arm]
                self.CN[j][arm] = self.CN[j][arm] + 1
                self.CQ[j][arm] = self.CQ[j][arm] + self.alpha * (R - self.CQ[j][arm])
                self.CR[i] = self.CR[i] + R/self.n

            if i % 100 == 0:
                print(i)


    def plot(self):
        plt.plot(self.VR, color='red', label='V')
        plt.plot(self.CR, color='blue', label='C')
        plt.legend()
        plt.show()


if __name__ == "__main__":
    r = NSG(0.1, 0.1, 10000, 500, 10)
    r.V()
    r.C()
    r.plot()
