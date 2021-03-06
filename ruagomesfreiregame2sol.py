import random

# LearningAgent to implement
# no knowledeg about the environment can be used
# the code should work even with another environment
class LearningAgent:

  # init
  # nS maximum number of states
  # nA maximum number of action per state
  def __init__(self, nS, nA):
    self.nS = nS
    self.nA = nA  
    # A matrix with Q_table[state][action] = Q(s, a)
    self.Q_table = [[] for state in range(nS)]

  def exploration_func(self, size_range, st):
    epsilon = 0.25
    if random.uniform(0, 1) < epsilon:
      return random.randint(0, size_range - 1) # explore more actions
    else:
      return self.Q_table[st].index(max(self.Q_table[st])) # explore best action

  # Select one action, used when learning  
  # st - is the current state        
  # aa - is the set of possible actions
  # for a given state they are always given in the same order
  # returns
  # a - the index to the action in aa
  def selectactiontolearn(self, st, aa):
    # print("select one action to learn better")
    
    # If this state doesnt have Q_table initialized 
    # Initialize with a list with the size of number of actions possible
    # For that State
    if(len(self.Q_table[st]) == 0):
      self.Q_table[st] = [0 for action in range(len(aa))]

    return self.exploration_func(len(aa), st)

  # Select one action, used when evaluating
  # st - is the current state        
  # aa - is the set of possible actions
  # for a given state they are always given in the same order
  # returns
  # a - the index to the action in aa
  def selectactiontoexecute(self, st, aa):
    # print("select one action to see if I learned")
    max_value = max(self.Q_table[st], default=None)
    if(max_value == None):
      return 0
    
    a = self.Q_table[st].index(max_value)
    return a

  # this function is called after every action
  # ost - original state
  # nst - next state
  # a - the index to the action taken
  # r - reward obtained
  def learn(self, ost, nst, a, r):
    #print("learn something from this data")
    learn_rate = 0.1
    gamma = 0.9

    self.Q_table[ost][a] = self.Q_table[ost][a] + learn_rate \
        * (r + gamma * max(self.Q_table[nst], default=0) - self.Q_table[ost][a])

