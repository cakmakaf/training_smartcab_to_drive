import random
import math
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
class LearningAgent(Agent):
    """ An agent that learns to drive in the Smartcab world.
        This is the object you will be modifying. """ 

    def __init__(self, env, learning=False, epsilon=1.0, alpha=0.5):
        super(LearningAgent, self).__init__(env)     # Set the agent in the evironment 
        self.planner = RoutePlanner(self.env, self)  # Create a route planner
        self.valid_actions = self.env.valid_actions  # The set of valid actions

        # Set parameters of the learning agent
        self.learning = learning # Whether the agent is expected to learn
        self.Q = dict()          # Create a Q-table which will be a dictionary of tuples
        self.epsilon = epsilon   # Random exploration factor
        self.alpha = alpha       # Learning factor

        # Set any additional class parameters as needed

        # set iter counter and random seed for random number generator
                    
        
        #the variable of iter was augmented 1 more after each iteration. We need to do this because epsilon will be reduced in every iteration.
        #Thus, we used the variable of iter and assigned 0 as its initial value.
        self.iter = 0
        #We should give a constant random seed in order to be able to allow pseudo random generator to generate the same random layout in each run. 
        #I gave the value of 3 to the seead of the random generator. Therefore, all generated random numbers will be the same in each run. 
        random.seed(3)

    def reset(self, destination=None, testing=False):
        """ The reset function is called at the beginning of each trial.
            'testing' is set to True if testing trials are being used
            once training trials have completed. """

        # Select the destination as the new location to route to
        self.planner.route_to(destination)
        
        ########### 
        ## TO DO ##
        ###########
        # Update epsilon using a decay function of your choice
        # Update additional class parameters as needed
        # If 'testing' is True, set epsilon and alpha to 0
        
        
        #If 'testing' is True, both epsilon and alpha should not change. However, If 'testing' is NOT True, the value of the epsilon must be reduced according to a determined decay function.  
        if testing:
            self.epsilon = 0.0
            self.alpha = 0.0
        else:
            #In this very spot, the iteration was increased and the decay function was applied to the epsilon.
            self.iter += 1.0 
            #self.epsilon = math.exp(-self.alpha*self.t)
            self.epsilon = math.fabs(math.cos(self.alpha*self.iter))

        return None

    def build_state(self):
        """ The build_state function is called when the agent requests data from the 
            environment. The next waypoint, the intersection inputs, and the deadline 
            are all features available to the agent. """


        # Collect data about the environment
        waypoint = self.planner.next_waypoint() # The next waypoint 
        inputs = self.env.sense(self)           # Visual input - intersection light and traffic
        deadline = self.env.get_deadline(self)  # Remaining deadline

        ########### 
        ## TO DO ##
        ###########
        
        # NOTE : you are not allowed to engineer features outside of the inputs available.
        # Because the aim of this project is to teach Reinforcement Learning, we have placed 
        # constraints in order for you to learn how to adjust epsilon and alpha, and thus learn about the balance between exploration and exploitation.
        # With the hand-engineered features, this learning process gets entirely negated.
        
        # Set 'state' as a tuple of relevant data for the agent    

           
        # we put all sensed values and waypoint into state tuple
        # since the remained time does not effect on choice, we neglected remaining time info    
        
        #tmp=inputs.values() 
        #tmp.append(waypoint)        
        
        #tmp=[inputs['light'],inputs['oncoming'],inputs['left'],inputs['right'], waypoint]
        #Once we set the inputs['right'] value the performens went down
        
        # During my analysis, I observed that the trainig is more successful if we set the 'state' variable with [light. oncoming, left, waypoint] inputs. 
        #Then I construct the following tuple. As you pay attention to 'right' input, you can see we didn't put it in the 'state' variable.
        #First, we created a list as 'tmp', and then we returned it after tronsformed it to a tuple.
        tmp=[inputs['light'],inputs['oncoming'],inputs['left'], waypoint]
        state=tuple(tmp)
        
        return state
        


    def get_maxQ(self, state):
        """ The get_maxQ function is called when the agent is asked to find the
            maximum Q-value of all actions based on the 'state' the smartcab is in. """

        ########### 
        ## TO DO ##
        ###########
        # Calculate the maximum Q-value of all actions for a given state
       
        #The Q-table has 4 colomns as "light", "oncoming", "left" and "waypoint". In this "maxQ" function, we compute the maximum of these 4 different value that was given for the state. 
        #First, we gave an extremely negative number to maxQ. Then, we loop it for each action to search the maximum value. 
        #The found max value was returned.       
        maxQ = -100000.0
        for action in self.Q[state]:
            if maxQ < self.Q[state][action]:
                maxQ = self.Q[state][action]
        return maxQ


    def createQ(self, state):
        """ The createQ function is called when a state is generated by the agent. """

        ########### 
        ## TO DO ##
        ###########
        # When learning, check if the 'state' is not in the Q-table
        # If it is not, create a new dictionary for that state
        #   Then, for each action available, set the initial Q-value to 0.0

        
        #In learning period, if a generated state is not in the Q-table, we add this state to the Q-table. 
        #As you can see, we define a dictionary with for entries with the value of 0.0 to add it to the Q-table.
        #Next, we add this dictionary to state index of the Q-table. We don't have to create an empty space since Q-table is a dictionary. 
        #We could consider the state as an index and then set the listed value directly in it.  
        #If the current state is in the table or we are in testing period, then we do not have to do anyhing. 

        if self.learning and not state in self.Q:
            self.Q[state] = {None:0.0, 'forward':0.0, 'left':0.0, 'right':0.0}

        return


    def choose_action(self, state):
        """ The choose_action function is called when the agent is asked to choose
            which action to take, based on the 'state' the smartcab is in. """

        # Set the agent state and default action
        self.state = state
        self.next_waypoint = self.planner.next_waypoint()
        action = None

        ########### 
        ## TO DO ##
        ###########
        # When not learning, choose a random action
        # When learning, choose a random action with 'epsilon' probability
        # Otherwise, choose an action with the highest Q-value for the current state
        # Be sure that when choosing an action with highest Q-value that you randomly select between actions that "tie".
        

        
        #The first if works for the following purpose: If the learning is not algorithmically, we need to return a random action among the existing actions. 
        if not self.learning:
            action = random.choice(self.valid_actions)
        else:
            #If the learning is epsilon-wise, then we pick a random action again. Therefore, we sometimes let to move randomly during the learning period. 
            if self.epsilon > 0.01 and self.epsilon > random.random():
                action = random.choice(self.valid_actions)
            else:            
                #We choose the best action with the probability of epsilon. To do this, the agent must search the maximum Q-value from the table and choose the particular action that gives this maxQ-value. 
                #If the maximum value is given at multiple actions, then the agent cose one of them randomly. 
                valid_actions = []
                maxQ = self.get_maxQ(state)
                for act in self.Q[state]:
                    if maxQ == self.Q[state][act]:
                        valid_actions.append(act)
                action = random.choice(valid_actions)

        return action


    def learn(self, state, action, reward):
        """ The learn function is called after the agent completes an action and
            receives a reward. This function does not consider future rewards 
            when conducting learning. """

        ########### 
        ## TO DO ##
        ###########
        # When learning, implement the value iteration update rule
        #   Use only the learning rate 'alpha' (do not use the discount factor 'gamma')

        
        #This is the spot that we update the Q-table acording to reward that was obtained when we make an action related with a specific state in the Q-table. 
        #Then, we update the Q-table acording to reinforcement learning algorithm. 
        if self.learning:
            self.Q[state][action] = self.Q[state][action] + self.alpha*(reward-self.Q[state][action])

        return


    def update(self):
        """ The update function is called when a time step is completed in the 
            environment for a given trial. This function will build the agent
            state, choose an action, receive a reward, and learn if enabled. """

        state = self.build_state()          # Get current state
        self.createQ(state)                 # Create 'state' in Q-table
        action = self.choose_action(state)  # Choose an action
        reward = self.env.act(self, action) # Receive a reward
        self.learn(state, action, reward)   # Q-learn

        return
        

def run():
    """ Driving function for running the simulation. 
        Press ESC to close the simulation, or [SPACE] to pause the simulation. """

    ##############
    # Create the environment
    # Flags:
    #   verbose     - set to True to display additional output from the simulation
    #   num_dummies - discrete number of dummy agents in the environment, default is 100
    #   grid_size   - discrete number of intersections (columns, rows), default is (8, 6)
    env = Environment()
    
    ##############
    # Create the driving agent
    # Flags:
    #   learning   - set to True to force the driving agent to use Q-learning
    #    * epsilon - continuous value for the exploration factor, default is 1
    #    * alpha   - continuous value for the learning rate, default is 0.5
    agent = env.create_agent(LearningAgent, learning=True, epsilon=1.0, alpha=0.01)
    
    ##############
    # Follow the driving agent
    # Flags:
    #   enforce_deadline - set to True to enforce a deadline metric
    env.set_primary_agent(agent, enforce_deadline=True)

    ##############
    # Create the simulation
    # Flags:
    #   update_delay - continuous time (in seconds) between actions, default is 2.0 seconds
    #   display      - set to False to disable the GUI if PyGame is enabled
    #   log_metrics  - set to True to log trial and simulation results to /logs
    #   optimized    - set to True to change the default log file name
    sim = Simulator(env, update_delay=0.01, log_metrics=True, optimized=True)
    
    ##############
    # Run the simulator
    # Flags:
    #   tolerance  - epsilon tolerance before beginning testing, default is 0.05 
    #   n_test     - discrete number of testing trials to perform, default is 0
    sim.run(n_test=100, tolerance=0.001)
    
    #Here we measure the length of the matrix, that been asked as Question 5 in the smartcab.ipynb file
    print(len(agent.Q))


if __name__ == '__main__':
    run()