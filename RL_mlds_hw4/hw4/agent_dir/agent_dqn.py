from agent_dir.agent import Agent
from collections import namedtuple
import numpy as np
import random
import torch

Transition = namedtuple('Transition',
                        ('state', 'action','reward', 'next_state'))

class Q_pi(torch.nn.Module):
    """
    Model : Q^pi (Q is critic, pi is actor)
        Input : Image 
        Output : value of each action
    """    
    def __init__(self, D_in, action_space_n, device):
        super(Q_pi, self).__init__()

        self.device = device
        self.epsilon = 0.5
        self.action_space_size = action_space_n
        H, D_out = 256, action_space_n
        self.model = torch.nn.Sequential(
            torch.nn.Linear(D_in, H),
            torch.nn.ReLU(),
            torch.nn.Linear(H, D_out), # output layer size = action space size
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)

    def flatten(self, x):
        """
        Flatten image for FC layers
            Input : Image (N,84,84,4)
            Output : torch.tensor : (N,84*84*4)
        """
        x = x.reshape(-1)
        x = np.expand_dims(x, axis=0)
        x = torch.from_numpy(x).float()
        return x

    def forward(self, x): # with epsilon greedy
        """
        Feedforward + epsilon greedy
            Input : torch.tensor : (N,84*84*4)
            Output : Q_fn(s,a) 
        """
        pred = self.model(self.flatten(x))
        return pred
    def epsilon_greedy(self, pred):
        """
        Output : epsilon_greedy(argmax(Q_fn(x)),random(action_space)) 
        """
        pred = pred.detach().numpy() # detach
        pred = pred.reshape(-1)
        if random.random() > self.epsilon:
            a = np.argmax(pred)
        else :
            a = np.random.choice(range(self.action_space_size))
        return a


class ReplayBuffer(object):
    """
    Replay Buffer for Q function
        default size : 10000 of (s_t, a_t, r_t, s_t+1)
    """
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """
        Push (s_t, a_t, r_t, s_t+1) into buffer
            Input : s_t, a_t, r_t, s_t+1
            Output : None
        """
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class Agent_DQN(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """

        super(Agent_DQN,self).__init__(env)

        if args.test_dqn:
            #you can load your model here
            print('loading trained model')
        ##################
        # YOUR CODE HERE #
        self.device = torch.device('cpu')
        print("Environment info:")
        print("  action space",self.env.get_action_space())
        print("  action space meaning:", self.env.get_meaning())
        print("  obeservation space",self.env.get_observation_space())

        self.Q_fn = Q_pi(84*84*4,self.env.get_action_space().n, self.device)
        self.Q_hat_fn = Q_pi(84*84*4,self.env.get_action_space().n, self.device)
        self.replay_buffer = ReplayBuffer() 
        self.BATCH_SIZE = 2

        ##################

    def init_game_setting(self):
        """

        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary

        """
        ##################
        # YOUR CODE HERE #
        ##################
        pass


    def train(self):
        """
        Implement your training algorithm here
        """
        ##################
        # YOUR CODE HERE #
        ep = 2
        for episode in range(ep):
            s_0 = self.env.reset()


            a_0 = self.make_action(s_0) # select action using epsilon greedy
            s_1, r_0, done, info =  self.env.step(a_0)
            
            # push data into replay buffer
            self.replay_buffer.push(s_0, a_0, r_0, s_1)
            
            # move to next state
            s_0 = s_1

            # optimize model
            self.optimize_model()




        ##################
        pass

    def optimize_model(self):
        if len(self.replay_buffer) < self.BATCH_SIZE:
            return

        """Sample"""
        # sample a batch from replay buffer
        transitions = self.replay_buffer.sample(self.BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=self.device, dtype=torch.uint8)
        print("not final mask", non_final_mask)
        
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        """Q Network"""
        # self.Q_fn(state_batch) return (N,(action space))
        # .gater will select the right index
        # reference : https://www.cnblogs.com/HongjianChen/p/9451526.html
        state_action_values = self.Q_fn(state_batch).gather(1, action_batch)


        """Q_hat(Target) Network"""
        next_state_values = torch.zeros(self.BATCH_SIZE, device=self.device)
        next_state_values[non_final_mask] = self.Q_hat_fn(non_final_next_states).max(1)[0].detach()


        """Regression"""


    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent

        Input:
            observation: np.array
                stack 4 last preprocessed frames, shape: (84, 84, 4)

        Return:
            action: int
                the predicted action from trained model
        """
        ##################
        # YOUR CODE HERE #
        Q_s_a = self.Q_fn.forward(observation)
        a_0 = self.Q_fn.epsilon_greedy(Q_s_a)
        return a_0 
        ##################
        
        
        
        
        
        #return self.env.get_random_action()

    def exploration_update(self):
        pass