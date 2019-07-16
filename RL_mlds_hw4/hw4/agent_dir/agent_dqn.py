from agent_dir.agent import Agent
from collections import namedtuple
import numpy as np
import random
import torch
import cv2
import matplotlib.pyplot as plt
Transition = namedtuple('Transition',
                        ('state', 'action','reward', 'next_state'))
def expand_dim(x):
    y = torch.unsqueeze(input=x,dim=0)
    return y

class Q_pi(torch.nn.Module):
    """
    Model : Q^pi (Q is critic, pi is actor)
        Input : Image 
        Output : value of each action
    """    
    def __init__(self, D_in, action_space_n, device):
        super(Q_pi, self).__init__()

        self.device = device
        self.epsilon = 1.0
        self.action_space_size = action_space_n
        H, D_out = 256, action_space_n
        self.model = torch.nn.Sequential(
            torch.nn.Linear(D_in, H),
            torch.nn.ReLU(),
            torch.nn.Linear(H, D_out), # output layer size = action space size
        ).to(self.device)
        self.optimizer = torch.optim.RMSprop(self.model.parameters(),lr=1.5e-4)

    def flatten(self, x):
        """
        Flatten image for FC layers
            Input : Image (N,84,84,4)
            Output : torch.tensor : (N,84*84*4)
        """
        N = x.shape[0] # read in N, C, H, W
        return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image

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
        self.Q_hat_fn.load_state_dict(self.Q_fn.state_dict())
        self.Q_hat_fn.eval()
        self.replay_buffer = ReplayBuffer() 
        self.BATCH_SIZE = 32

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
        NUM_EPISODES = 10
        TARGET_UPDATE_C = 1000
        UPDATE_FREQUENCY = 4
        DEBUG_COUNT = 0
        for episode in range(NUM_EPISODES):
            s_0 = torch.from_numpy(self.env.reset())
            episode_reward = 0
            while(True):
                self.Q_fn.eval()
                a_0 = self.make_action(s_0) # select action using epsilon greedy
                s_1, r_0, done, info =  self.env.step(a_0)
                episode_reward += r_0
                DEBUG_COUNT = self.debug_observation_frame(DEBUG_COUNT,s_0)
                
                if done == True:
                    break
                # push data into replay buffer
                # note : 
                # 1. Before pushing, transform into torch.tensor
                # 2. expand_dim : x.unsqueeze_(0).shape = (1,x.shape)
                a_0 = torch.tensor([a_0], device=self.device)
                r_0 = torch.tensor([r_0], device=self.device)
                s_1 = torch.from_numpy(s_1)
                self.replay_buffer.push(expand_dim(s_0), expand_dim(a_0), expand_dim(r_0), expand_dim(s_1))
                # move to next state
                s_0 = s_1

                # optimize Q model
                if len(self.replay_buffer) % UPDATE_FREQUENCY == 0:        
                    self.optimize_model()
                
                # update  Q' model
                if episode % TARGET_UPDATE_C == 0:
                    self.Q_hat_fn.load_state_dict(self.Q_fn.state_dict())
            
            self.Q_fn.epsilon = self.epsilon_decline(episode + 1, NUM_EPISODES)
            print("\rEpisode Reward: {:.2f}".format(episode_reward, end=""))
        ##################
        

    def optimize_model(self):
        if len(self.replay_buffer) < self.BATCH_SIZE:
            return 

        self.Q_fn.train()
        self.Q_hat_fn.eval()
        """Sample from Replay Buffer"""
        # sample a batch from replay buffer
        transitions = self.replay_buffer.sample(self.BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=self.device, dtype=torch.uint8)
        
        
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
        # .max(1,keepdim=True)[0] return (N,1) vector 
        # same as below 
        # for data in range(batch):
        #   next_state_values[data] = argmax Q'(s_t+1, r_t)
        next_state_values = self.Q_hat_fn(non_final_next_states).max(1,keepdim=True)[0].detach()
        expected_state_action_values = (next_state_values) + reward_batch

        """Regression"""
        loss_fn = torch.nn.MSELoss()
        loss = loss_fn(state_action_values, expected_state_action_values)

        # update
        self.Q_fn.optimizer.zero_grad()
        loss.backward()
        self.Q_fn.optimizer.step()


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
        Q_s_a = self.Q_fn.forward(expand_dim(observation))
        a_0 = self.Q_fn.epsilon_greedy(Q_s_a)
        return a_0 
        ##################        

    def epsilon_decline(self, ep, NUM_EPISODES):
        eps_threshold = 0.025 + ((1 - 0.025)*(ep / NUM_EPISODES))
        return eps_threshold 

    def debug_observation_frame(self, count, observation):
        '''
        Input :
            count : means each frame of the postfix image file name 
            action : test the action meaning

        Output :
            debug_image/ : directory of one epoch images
            debug_action.txt : action text file
        '''
        #img = np.float32(observation[:,:,0].numpy())
        #cv2.imwrite("debug_image/"'output'+str(count)+str('.png'), img)
        #count+=1
        #for i in range(4):
        #    cv2.imwrite("debug_image/"'output'+str(count)+str('.png'), observation[:,:,i].numpy())
        #    count += 1
        return count
        