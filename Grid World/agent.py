import numpy as np
from collections import deque
from abc import ABCMeta, abstractmethod
from environment import Environment, WIN, LOSE, ACTION_MOVE


ACTION_SPACE = ('North', 'South', 'East', 'West')
STATE_MAP = {
            (0,0):0, (0,1):1, (0,2):2, (0,3):3,
            (1,0):4, (1,1):5, (1,2):6, (1,3):7,
            (2,0):8, (2,1):9, (2,2):10, (2,3):11
            }
ACTION_MAP = {'North':0, 'South':1, 'East':2, 'West':3}
REVERSE_ACTION_MAP = {0:'North', 1:'South', 2:'East', 3:'West'}


class Agent(metaclass=ABCMeta):
    def __init__(self, stochastic=True):
        self.env = Environment(stochastic)

    @abstractmethod
    def decide_action(self):
        pass

    @abstractmethod
    def run(self):
        pass


class RandomAgent(Agent):
    '''
    Agent that selects action randomly
    '''
    def decide_action(self):
        action = np.random.choice(ACTION_SPACE)
        return action
    
    def run(self, n_iter, show_board=False):
        result_hist = []
        for iter in range(n_iter):
            while not (self.env.current_state in (WIN, LOSE)):    # termination condition
                action = self.decide_action()
                next_state = self.env.give_next_state(action)
                reward = self.env.give_reward(next_state)
                if show_board:
                    self.env.show_board(self.env.current_state, next_state)
                self.env.update_state(next_state)
            result_hist.append(self.env.cum_reward)
            self.env = Environment()    # reset Environment for every iteration
        return result_hist


class MCAgent(Agent):
    '''
    Agent using Monte Carlo method to estimate value functions when dynamics are unknown
    '''
    def est_av_func(self, n_samp, eps, gamma):
        '''
        Monte Carlo estimation of action-value function

        n_samp: number of Monte Carlo episodes
        eps: epsilon value for epsilon-greedy algorithm
        gamma: discount rate for reward
        '''
        cum_av_func = np.zeros([12,4])  # state x action
        count = np.zeros([12,4])        # state x action
        _, trajectory = self.run(n_samp, eps)   # trajectory is sampled by decide_action and environment
        for episode in trajectory:
            av_func = np.zeros([12,4])  # state x action
            est_reward = 0
            episode.reverse()
            for state, action, reward in episode:
                est_reward = gamma * est_reward + reward
                av_func[STATE_MAP[state], ACTION_MAP[action]] = est_reward
            count += np.where(av_func != 0, 1, 0)
            cum_av_func += av_func
        mean_av_func = np.where(count == 0, 0, cum_av_func / count)
        return mean_av_func
    
    def update_av_func(self, av_func):
        self.av_func = av_func

    def decide_action(self, state, eps):
        '''
        epsilon-greedy algorithm that returns action for given state

        state: current state
        eps: epsilon value for epsilon-greedy algorithm
        '''
        if np.random.random() >= eps:
            # greedy action with prob. 1 - epsilon
            action = REVERSE_ACTION_MAP[np.argmax(self.av_func[STATE_MAP[state],:])]
        else:
            # random action with prob. epsilon                 
            action = np.random.choice(ACTION_SPACE)
        return action

    def train(self, n_samp=10000, gamma=0.9, eps=0.1, eps_decay=True, eps_decay_rate=0.99, tol=0.05):
        '''
        n_samp: number of Monte Carlo episodes
        gamma: discount rate for reward
        eps: epsilon value for epsilon-greedy algorithm
        eps_decay: whether to decay epsilon
        eps_decay_rate: decay rate of epsilon for every iteration
        tol: threshold for stopping training
        '''
        iter = 1
        self.av_func = np.zeros([12,4])
        av_func = self.est_av_func(n_samp, eps=1, gamma=gamma)  # initial policy is random policy(eps = 1)
        while True:
            print('iteration:', iter)
            print('old action-value function:', self.av_func)
            print('updated action-value function:', av_func)
            print(abs(self.av_func - av_func) < tol)
            print('-' * 100)

            # stop when difference between all values of a-v function and updated a-v function is below tol
            if (abs(self.av_func - av_func) < tol).all():
                print('Converged to optimal action-value function!')
                break

            self.update_av_func(av_func)
            av_func = self.est_av_func(n_samp, eps, gamma)

            if eps_decay:
                eps *= eps_decay_rate
            iter += 1

    def run(self, n_iter, eps = 0.1, show_board=False):
        trajectory_list = []
        result_hist = []
        for iter in range(n_iter):
            trajectory = deque()
            while not (self.env.current_state in (WIN, LOSE)):    # termination condition
                action = self.decide_action(self.env.current_state, eps=eps)
                next_state = self.env.give_next_state(action)
                reward = self.env.give_reward(next_state)
                trajectory.append((self.env.current_state, action, reward))
                if show_board:
                    self.env.show_board(self.env.current_state, next_state)
                self.env.update_state(next_state)
            trajectory_list.append(trajectory)
            result_hist.append(self.env.cum_reward)
            self.env = Environment()    # reset Environment for every iteration
        return result_hist, trajectory_list