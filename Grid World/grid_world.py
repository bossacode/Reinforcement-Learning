import numpy as np

ACTION_SPACE = ('North', 'South', 'East', 'West')
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
    ACTION_MAP = {'North':(1,0), 'South':(-1,0), 'East':(0,1), 'West':(0,-1)}
    
    def __init__(self):
        self.current_state = START
        self.reward = 0
        self.dashboard = np.array([[0, 0, 0, 'WIN'],
                                   [0, 'BLOCK', 0, 'LOSE'],
                                   ['*', 0, 0, 0]])

    @staticmethod
    def check_valid_state(state):
        if state in STATE_SPACE and state != BLOCK:
            return True
        else:
            return False

    def give_next_state(self, action, show_board = False):
        next_state = tuple([i + j for i, j in zip(self.current_state, self.__class__.ACTION_MAP[action])])
        if self.check_valid_state(next_state) == True:
            if show_board:
                self.dashboard[self.current_state] = 0
                self.dashboard[next_state] = '*'
                print(self.dashboard)
                print('-' * 50)
            self.current_state = next_state
        else:
            if show_board:
                print(self.dashboard)
                print('-' * 50)
        
    def give_reward(self):
        if self.current_state == WIN:
            self.reward += 1
        elif self.current_state == LOSE:
            self.reward -= 1
        else:
            pass


class RandomAgent:
    def __init__(self):
        self.env = Environment()

    @staticmethod
    def decide_action(state):
        action = np.random.choice(ACTION_SPACE)
        if action == 'North':
            action = np.random.choice(('North', 'East', 'West'), p=(0.8, 0.1, 0.1))
        return action
    
    def run(self, n_iter, show_board=False):
        history = []
        for iter in range(n_iter):
            while not (self.env.current_state in (WIN, LOSE)):    # termination condition
                action = self.decide_action(self.env.current_state)
                self.env.give_next_state(action, show_board)    
                self.env.give_reward()
            history.append(self.env.reward)

            self.env = Environment()    # reset for every iteration
        return history



if __name__ == '__main__':
    ra = RandomAgent()
    ra.run(n_iter = 1, show_board = True)