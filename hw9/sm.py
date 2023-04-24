from util import *
import numpy as np

class SM:
    start_state = None  # default start state

    def transition_fn(self, s, x):
        '''s:       the current state
           x:       the given input
           returns: the next state'''
        raise NotImplementedError

    def output_fn(self, s):
        '''s:       the current state
           returns: the corresponding output'''
        raise NotImplementedError

    def transduce(self, input_seq):
        '''input_seq: the given list of inputs
           returns:   list of outputs given the inputs'''
        # Your code here
        outputs = []
        state = self.start_state
        for inp in input_seq:
            state = self.transition_fn(state, inp)
            outputs.append(self.output_fn(state))
        return outputs

class Accumulator(SM):
    start_state = 0

    def transition_fn(self, s, x):
        return s + x

    def output_fn(self, s):
        return s


class Binary_Addition(SM):
    start_state = (0, 0) # Change

    def transition_fn(self, s, x):
        # Your code here
        add = s[1] + x[0] + x[1]
        carry = 1 if add > 1 else 0
        return (add % 2, carry)  

    def output_fn(self, s):
        # Your code here
        return s[0]


class Reverser(SM):
    start_state = [[], [], False]

    def transition_fn(self, s, x):
        # Your code here
        
        toret = None
        
        if s[2] == False:
            if x == 'end':
                # Get last word in state list
                toret = s[1][-1]
                # Remove word from state list
                del s[1][-1]
                # Set end state as True
                s[2] = True
            else:
                # Append to state word from setence to reverse
                s[1].append(x)
        else:
            if len(s[1]) > 0:
                toret = s[1][-1]
                del s[1][-1]
        
        s[0].append(toret)
        
        return [s[0], s[1], s[2]]

    def output_fn(self, s):
        # Your code here
        return s[0][-1]
    

class RNN(SM):
    def __init__(self, Wsx, Wss, Wo, Wss_0, Wo_0, f1, f2):
        self.start_state = np.zeros((Wss.shape[0], 1))
        self.Wsx = Wsx
        self.Wss = Wss
        self.Wo = Wo
        self.Wss_0 = Wss_0
        self.Wo_0 = Wo_0
        self.f1 = f1
        self.f2 = f2

    def transition_fn(self, s, i):
        return self.f1(np.dot(self.Wss, s) + np.dot(self.Wsx, i) + self.Wss_0)

    def output_fn(self, s):
        return self.f2(np.dot(self.Wo, s) + self.Wo_0)
