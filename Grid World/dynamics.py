# dynamics outputs probabilty for (next state, reward) given (current state, action)
# number of states: 12  {
                        # 0:(0,0), 1:(0,1), 2:(0,2), 3:(0,3),
                        # 4:(1,0), 5:(1,1), 6:(1,2), 7:(1,3),
                        # 8:(2,0), 9:(2,1), 10:(2,2), 11:(2,3),
                        # }
# number of rewards: 3  {0:0, 1:1, 2:-1}
# number of action: 4   {0:North, 1:South, 2:East, 3:West}
# dynamics will be set as 4D tensor in the shape of (12(current state) x 4(action) x 12(next state) x 3(reward))

import numpy as np

def cal_dynamcis(stochastic=True):
    dynamics = np.zeros([12, 4, 12, 3]) # current state x action x next state x reward
    for current_state in range(12):
        if current_state in [3,5,7]:    # BLOCK, WIN, LOSE
            continue
        
        # action = North
        if current_state // 4 == 0 or current_state == 9:     # top row of 3 x 4 state space or BLOCK is above(state (2,1))
            dynamics[current_state, 0, current_state, 0] += 0.8 if stochastic else 1
        elif current_state == 11:       # state (2,3) -> LOSE is above
            dynamics[current_state, 0, current_state-4, 2] += 0.8 if stochastic else 1
        else:
            dynamics[current_state, 0, current_state-4, 0] += 0.8 if stochastic else 1

        # action = East
        if current_state % 4 == 3 or current_state == 4:  # right-most column of 3 x 4 state space or BLOCK is on the right(state (1,0))
            dynamics[current_state, 2, current_state, 0] += 1
            if stochastic:
                dynamics[current_state, 0, current_state, 0] += 0.1 # stochastic movement for North
        elif current_state == 2:    # state (0,2) -> WIN is on the right
            dynamics[current_state, 2, current_state+1, 1] += 1
            if stochastic:
                dynamics[current_state, 0, current_state+1, 1] += 0.1 # stochastic movement for North
        elif current_state == 6:    # state (1,2) -> LOSE is on the right
            dynamics[current_state, 2, current_state+1, 2] += 1
            if stochastic:
                dynamics[current_state, 0, current_state+1, 2] += 0.1 # stochastic movement for North
        else:
            dynamics[current_state, 2, current_state+1, 0] += 1
            if stochastic:
                dynamics[current_state, 0, current_state+1, 0] += 0.1 # stochastic movement for North

        # action = West
        if current_state % 4 == 0 or current_state == 6:  # left-most column of 3 x 4 state space or BLOCK is on the left(state (1,2))
            dynamics[current_state, 3, current_state, 0] += 1
            if stochastic: 
                dynamics[current_state, 0, current_state, 0] += 0.1 # stochastic movement for North
        else:
            dynamics[current_state, 3, current_state-1, 0] += 1
            if stochastic:
                dynamics[current_state, 0, current_state-1, 0] += 0.1 # stochastic movement for North

        # action = South
        if current_state // 4 == 2 or current_state == 1:     # bottom row of 3 x 4 state space or BLOCK is below(state (0,1))
            dynamics[current_state, 1, current_state, 0] +=  1
        else:
            dynamics[current_state, 1, current_state+4, 0] += 1

    return dynamics
