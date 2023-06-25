import numpy as np


ACTION_MOVE = {'North':(-1,0), 'South':(1,0), 'East':(0,1), 'West':(0,-1)}
STATE_SPACE = (
    (0,0), (0,1), (0,2), (0,3),
    (1,0), (1,1), (1,2), (1,3),
    (2,0), (2,1), (2,2), (2,3),
)
START = (2,0)
BLOCK = (1,1)
WIN = (0,3)
LOSE = (1,3)


class Environment:
    def __init__(self, stochastic=True):
        self.current_state = START
        self.cum_reward = 0
        self.dashboard = np.array([[0, 0, 0, 'WIN'],
                                   [0, 'BLOCK', 0, 'LOSE'],
                                   ['*', 0, 0, 0]])
        self.stochastic = stochastic

    @staticmethod
    def check_valid_state(state):
        if state in STATE_SPACE and state != BLOCK:
            return True
        else:
            return False

    def give_next_state(self, action):
        if self.stochastic and action == 'North':
            action = np.random.choice(('North', 'East', 'West'), p=(0.8, 0.1, 0.1))

        next_state = tuple([i + j for i, j in zip(self.current_state, ACTION_MOVE[action])])

        if self.check_valid_state(next_state):
            return next_state
        else:
            return self.current_state

    def give_reward(self, state):
        if state == WIN:
            self.cum_reward += 1
            return 1
        elif state == LOSE:
            self.cum_reward -= 1
            return -1
        else:
            return 0
        
    def update_state(self, next_state):
        self.current_state = next_state

    def show_board(self, current_state, next_state):
        self.dashboard[current_state] = 0
        self.dashboard[next_state] = '*'
        print(self.dashboard)
        print('-' * 50)