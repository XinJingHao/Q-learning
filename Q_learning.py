import numpy as np

class QLearningAgent(object):
    def __init__(self, env_with_dw, s_dim, a_dim, lr=0.01, gamma=0.9, exp_noise=0.1):
        self.env_with_dw = env_with_dw
        self.a_dim = a_dim
        self.lr = lr
        self.gamma = gamma
        self.epsilon = exp_noise
        self.Q = np.zeros((s_dim, a_dim))

    def select_action(self,s):
        '''e-greedy policy'''
        if np.random.uniform(0, 1) < self.epsilon:
            a = np.random.choice(self.a_dim)
        else:
            a = self.predict(s)
        return a

    def predict(self, s):
        '''Deterministic policy'''
        Q_s = self.Q[s, :]
        maxQ = np.max(Q_s)
        action_list = np.where(Q_s == maxQ)[0]
        a = np.random.choice(action_list)
        return a


    # Update Q table
    def train(self, s, a, r, s_, done):
        Q_sa = self.Q[s, a]
        if self.env_with_dw:
            target_Q = r + (1-done)*self.gamma * np.max(self.Q[s_, :])
        else:
            target_Q = r + self.gamma * np.max(self.Q[s_, :])
        self.Q[s, a] += self.lr * (target_Q - Q_sa)


    #save Q table
    def save(self):
        npy_file = 'model/q_table.npy'
        np.save(npy_file, self.Q)
        print(npy_file + ' saved.')

    #load Q table
    def restore(self, npy_file='model/q_table.npy'):
        self.Q = np.load(npy_file)
        print(npy_file + ' loaded.')
