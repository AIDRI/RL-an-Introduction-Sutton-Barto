import numpy as np

#######################################
# Test Bed from Sutton and Barto book #
#######################################

class EnvRunner:
    def __init__(self):
        self.qstara = [0]*10
        for i in range(10):
            self.qstara[i] = np.random.standard_normal()

    def play(self, arm):
        return np.random.normal(self.qstara[arm], 1)

##########################################
# Here we have a simple greedy algorithm #
##########################################

class GreedyAlg: # that s the worst alg i ve ever seen :joy:
    def __init__(self, k, arms):
        self.env = EnvRunner()
        self.k = k
        self.arms = arms
        self.Q = [0]*arms
        self.N = [0]*arms
        self.R = [0]*k
        self.results = []

    def play(self):
        for t in range(1000):
            for i in range(self.k):
                arm = np.argmax(self.Q)
                self.R[i] = self.env.play(arm)
                self.N[arm] += 1
                self.Q[arm] = self.Q[arm] + (1/self.N[arm])*(self.R[i] - self.Q[arm])

                self.results.append(self.R)
                self.Q = [0]*self.arms
                self.N = [0]*self.arms
                self.R = [0]*self.k

                if t%100==0:
                    print(f"{t}th iteration")

        return self.results

#############################################
# And now, let's code an e-greedy algorithm #
#############################################

class EGreedyAlg:
    def __init__(self, epsilon, k, arms):
        self.epsilon = epsilon
        self.env = EnvRunner()
        self.k = k
        self.arms = arms
        self.Q = [0]*arms
        self.N = [0]*arms
        self.R = [0]*k
        self.results = []

    def play(self):
        for t in range(1000):
            for i in range(self.k):
                if np.random.random() < self.epsilon:
                    arm = np.random.randint(self.arms)
                else:
                    arm = np.argmax(self.Q)
                self.R[i] = self.env.play(arm)
                self.N[arm] += 1
                self.Q[arm] = self.Q[arm] + (1/self.N[arm])*(self.R[i] - self.Q[arm])

                self.results.append(self.R)
                self.Q = [0]*self.arms
                self.N = [0]*self.arms
                self.R = [0]*self.k

                if t%100==0:
                    print(f"{t}th iteration")

        return self.results


####################
# U.C.B. algorithm #
####################

class UCBAlg:
    def __init__(self, c, k, arms):
        self.c = c
        self.env = EnvRunner()
        self.k = k
        self.arms = arms
        self.Q = [0]*arms
        self.N = [0]*arms
        self.R = [0]*k
        self.results = []

    def play(self):
        for t in range(1000):
            for i in range(self.k):
                arm = np.argmax(self.Q + self.c*np.sqrt(np.log(i+1)/self.N))
                self.R[i] = self.env.play(arm)
                self.N[arm] += 1
                self.Q[arm] = self.Q[arm] + (1/self.N[arm])*(self.R[i] - self.Q[arm])

            self.results.append(self.R)
            self.Q = [0]*self.arms
            self.N = [0]*self.arms
            self.R = [0]*self.k

            if t%100==0:
                print(f"{t}th iteration")

        return self.results



########
# test #
########

import matplotlib.pyplot as plt

#alg = GreedyAlg(k=1000, arms=10)
#alg = EGreedyAlg(epsilon=0.1, k=1000, arms=10)
alg = UCBAlg(c=2, k=1000, arms=10)

results = alg.play()

fin = []
for i in range(len(results[0])):
    tmp = 0
    for j in range(len(results)):
        tmp += results[j][i]
    fin.append(tmp/len(results))


plt.plot(fin)
plt.show()
