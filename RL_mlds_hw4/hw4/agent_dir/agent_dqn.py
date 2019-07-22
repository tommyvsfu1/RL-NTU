from agent_dir.agent import Agent
from collections import namedtuple
import numpy as np
import random
import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F  # useful stateless functions
from logger import TensorboardLogger



SEED = 11037
random.seed(SEED)
np.random.seed(SEED)


Transition = namedtuple('Transition',
                        ('state', 'action','reward', 'next_state'))


def expand_dim(x):
    y = torch.unsqueeze(input=x,dim=0)
    return y

def prepro(o):
    o = o.transpose((2, 0, 1))
    return o


class Q_pi(torch.nn.Module):
    """
    Model : Q^pi 
        Input : Image 
        Output : value of each action
    """    
    def __init__(self, action_space_n, device):
        super(Q_pi, self).__init__()        
        # self.model = torch.nn.Sequential(
        #     torch.nn.Conv2d(4, )
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(H, D_out), # output layer size = action space size
        # ).to(self.device)
        self.conv1 = torch.nn.Conv2d(4,32,kernel_size=8,stride=[4,4])
        torch.nn.init.kaiming_normal_(self.conv1.weight)
        self.conv2 = torch.nn.Conv2d(32,64,kernel_size=4,stride=[2,2])
        torch.nn.init.kaiming_normal_(self.conv2.weight)
        self.conv3 = torch.nn.Conv2d(64,64,kernel_size=3,stride=[1,1])
        torch.nn.init.kaiming_normal_(self.conv3.weight)
        
        self.fc1_adv = torch.nn.Linear((3136), 512)
        self.fc1_val = torch.nn.Linear((3136), 512)
        self.fc2_adv = torch.nn.Linear((512), 4)
        self.fc2_val = torch.nn.Linear((512), 1)
        torch.nn.init.kaiming_normal_(self.fc1_adv.weight)
        torch.nn.init.kaiming_normal_(self.fc1_val.weight)
        torch.nn.init.kaiming_normal_(self.fc2_adv.weight)
        torch.nn.init.kaiming_normal_(self.fc2_val.weight)

    def flatten(self, x):
        """
        Flatten image for FC layers
            Input : Image (N,84,84,4)
            Output : torch.tensor : (N,84*84*4)
        """
        N = x.shape[0] # read in N, C, H, W
        return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image

    def forward(self, x): 
        """
        Feedforward 
            Input : torch.tensor : (N,84*84*4)
            Output : Q_fn(s,a) : (N,action_space_n=4)
        """
        # do not use softmax in DQN !!!!!
        # and remember don't use relu on last layers, since it somewhere means kill half of output values
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.flatten(x)
        adv = F.relu(self.fc1_adv(x))
        val = F.relu(self.fc1_val(x))

        adv = self.fc2_adv(adv)
        val = self.fc2_val(val).expand(x.size(0),4)
        x = val + adv - adv.mean(1).unsqueeze(1).expand(x.size(0),4)
        return x
        
    def epsilon_greedy(self, pred, epsilon, tensorboard):
        """
        Output : epsilon_greedy(argmax(Q_fn(x)),random(action_space)) 
        """
        pred = pred.detach() # detach
        pred = pred.reshape(-1)
        if random.random() > epsilon:
            a = torch.argmax(pred).item()
        else :
            a = np.random.choice(range(4))
        tensorboard.scalar_summary("make_action_action", a)
        return a


class ReplayBuffer(object):
    """
    Replay Buffer for Q function
        default size : 20000 of (s_t, a_t, r_t, s_t+1)
    """
    def __init__(self, capacity=20000):
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
        print("Hardware Device info")
        print("  using device:",self.device)
        print("Environment info:")
        print("  action space",self.env.get_action_space())
        print("  action space meaning:", self.env.get_meaning())
        print("  obeservation space",self.env.get_observation_space())

        self.env.seed(SEED)
        self.Q_fn = Q_pi(self.env.get_action_space().n, self.device).to(device=self.device)
        self.Q_hat_fn = Q_pi(self.env.get_action_space().n, self.device).to(device=self.device)
        self.Q_hat_fn.load_state_dict(self.Q_fn.state_dict())
        self.Q_hat_fn.eval()
        self.REPLAY_BUFFER_START_SIZE = 20000
        self.replay_buffer = ReplayBuffer(self.REPLAY_BUFFER_START_SIZE) 
        self.BATCH_SIZE = 32
        self.Q_epsilon = 1.0
        self.optimizer = torch.optim.RMSprop(self.Q_fn.parameters(),lr=1.5e-4)
        self.tensorboard = TensorboardLogger("./logger_dqn/exp1")
        """
        optimizer = torch.optim.RMSprop(self.Q_fn.parameters(),lr=1.5e-4)
        x = torch.zeros((10, 4, 84, 84), dtype=torch.float32)
        self.Q_fn.train()
         #x = prepro(np.zeros((84,84,4)))
        # #x = torch.from_numpy(x)
        # #x = expand_dim(x)
        scores = self.Q_fn(x)
        loss_fn = torch.nn.MSELoss()
        loss = loss_fn(scores, torch.zeros((10,4)))
        optimizer.zero_grad()

        # # This is the backwards pass: compute the gradient of the loss with
        # # respect to each  parameter of the model.
        loss.backward()

        # # Actually update the parameters of the model using the gradients
        # # computed by the backwards pass.
        optimizer.step()
        # print("scores shape", scores.shape)
        ##################
        """
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
        
        NUM_EPISODES = 5000
        TARGET_UPDATE_C = 1000
        UPDATE_FREQUENCY = 4
        DEBUG_COUNT = 0
        LINEAR_DECLINE_STEP = 100000
        MAX_STEP = 2000
        time_step = 0
        epsisode_history = []
        for episode in range(NUM_EPISODES):
            s_0 = torch.from_numpy(prepro(self.env.reset()))
            episode_reward = 0
            for _ in range(MAX_STEP):
                self.Q_fn.eval()
                time_step += 1
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
                s_1 = torch.from_numpy(prepro(s_1))
                self.replay_buffer.push(expand_dim(s_0), expand_dim(a_0), expand_dim(r_0), expand_dim(s_1))
                # move to next state
                s_0 = s_1

                # optimize Q model
                if (time_step % UPDATE_FREQUENCY) == 0 and len(self.replay_buffer) > self.REPLAY_BUFFER_START_SIZE:        
                    print("update")
                    self.optimize_model("DQN")
                
                # update  Q' model
                if time_step % TARGET_UPDATE_C == 0 and len(self.replay_buffer) > self.REPLAY_BUFFER_START_SIZE:
                    self.Q_hat_fn.load_state_dict(self.Q_fn.state_dict())
                
                self.Q_epsilon = self.epsilon_decline(time_step, LINEAR_DECLINE_STEP)
            epsisode_history.append(episode_reward)
            print("episode",episode,"average 100 reward",np.mean(epsisode_history[-100:]),"time_step",time_step,"epsilon",self.Q_epsilon)
            #print("\rEpisode Reward: {:.2f}".format(episode_reward, end=""))
        plt.plot(range(len(epsisode_history)), epsisode_history)
        plt.savefig('reward_history.png')
        ##################
        

    def optimize_model(self, improvment):
        if len(self.replay_buffer) < self.BATCH_SIZE:
            return 

        self.Q_fn.train()
        self.Q_hat_fn.eval()
        self.tensorboard.time_s += 1
        """Sample from Replay Buffer"""
        # sample a batch from replay buffer
        transitions = self.replay_buffer.sample(self.BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        state_batch = torch.cat(batch.state).to(self.device)
        action_batch = torch.cat(batch.action).to(self.device)
        reward_batch = torch.cat(batch.reward).to(self.device)
        next_state_batch = torch.cat(batch.state).to(self.device)


        """ Q Network """
        # self.Q_fn(state_batch) return (N,action space_n=4)
        # .gater will select the right index
        # reference : https://www.cnblogs.com/HongjianChen/p/9451526.html
        state_action_values = self.Q_fn(state_batch).gather(1, action_batch)
        self.tensorboard.histogram_summary("state_action_values", state_action_values)

        """ Q_hat(Target) Network """
        # .max(1,keepdim=True)[0] return (N,1) vector 
        # same as below 
        # for data in range(batch):
        #   next_state_values[data] = argmax Q'(s_t+1, r_t)
        if improvment == "DQN":
            next_state_values= self.Q_hat_fn(next_state_batch).max(1,keepdim=True)[0].detach()
            expected_state_action_values = 0.99 * (next_state_values) + reward_batch
            self.tensorboard.histogram_summary("next_state_values", next_state_values)
            self.tensorboard.histogram_summary("reward_batch", reward_batch)
            self.tensorboard.histogram_summary("expected_bellman", expected_state_action_values)
        elif improvment == "DDQN":
            select_action_values= self.Q_fn(next_state_batch).detach()
            #print("select action values", select_action_values)
            select_action = torch.argmax(select_action_values, dim = 1, keepdim = True)
            #print("select action", select_action)
            next_state_values, _ = self.Q_hat_fn(next_state_batch).detach()
            next_state_values = next_state_values.gather(1, select_action)
            expected_state_action_values = 0.999 * (next_state_values) + reward_batch


        """Regression"""
        loss_fn = torch.nn.MSELoss()
        loss = loss_fn(state_action_values, expected_state_action_values)
        self.tensorboard.scalar_summary("total_loss", loss.item())


        self.tensorboard.histogram_summary("q_conv1", self.Q_fn.conv1.weight)
        self.tensorboard.histogram_summary("q_conv2", self.Q_fn.conv2.weight)
        self.tensorboard.histogram_summary("q_conv3", self.Q_fn.conv3.weight)
        self.tensorboard.histogram_summary("q_fc1_adv", self.Q_fn.fc1_adv.weight)
        self.tensorboard.histogram_summary("q_fc2_adv", self.Q_fn.fc2_adv.weight)
        self.tensorboard.histogram_summary("q_fc1_val", self.Q_fn.fc1_val.weight)
        self.tensorboard.histogram_summary("q_fc2_val", self.Q_fn.fc2_val.weight)

        # update
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


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
        Q_s_a= self.Q_fn.forward(expand_dim(observation).to(self.device))
        a_0 = self.Q_fn.epsilon_greedy(Q_s_a, self.Q_epsilon, self.tensorboard)
        return a_0 
        ##################        

    def epsilon_decline(self, t, step):
        if t <= step:
            eps_threshold = 1 - (0.975 *(t / step))
        else :
            eps_threshold = 0.025
        
        if eps_threshold < 0.025:
            eps_threshold = 0.025
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
        
