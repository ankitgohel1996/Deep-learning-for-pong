import numpy as np
import nnet

class NAgent:
    def __init__(self, W, B, normalize):
        self.W = W
        self.B = B
        self.normalize = normalize
    
    def normal(self, state):
        for col in range(state.size):
            mean = self.normalize[col][0]
            sd = self.normalize[col][1]
            state[col] = (state[col] - mean)/sd
        return state
    
    def choose_action(self, state):
        state = self.normal(np.array(state))
        return (nnet.FourNetwork(state, self.W, self.B, None, True)[0]-1)
