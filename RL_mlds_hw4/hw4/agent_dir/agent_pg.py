from agent_dir.agent import Agent
import copy
import scipy
import scipy.misc
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import skimage
import torch.nn.functional as F  # useful stateless functions
from logger import TensorboardLogger

seed = 9487
np.random.seed(seed)
torch.manual_seed(seed)

def flatten(x):
    """
    Flatten image for FC layers
        Input : Image (N,84,84,4)
        Output : torch.tensor : (N,84*84*4)
    """
    N = x.shape[0] # read in N, C, H, W
    return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image

class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.values = []
    
    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.values[:]

class ActorCritic(torch.nn.Module):
    def __init__(self, state_dim, action_dim, n_latent_var):
        super(ActorCritic, self).__init__()    

        # CNN (feature layers)    
        self.conv1 = torch.nn.Conv2d(1,32,kernel_size=8,stride=[4,4],padding=0)
        torch.nn.init.kaiming_normal_(self.conv1.weight)
        self.conv2 = torch.nn.Conv2d(32,64,kernel_size=4,stride=[2,2],padding=0)
        torch.nn.init.kaiming_normal_(self.conv2.weight)
        self.conv3 = torch.nn.Conv2d(64,64,kernel_size=3,stride=[1,1],padding=0)
        torch.nn.init.kaiming_normal_(self.conv3.weight)
        
        # actor
        self.fc4 = torch.nn.Linear((2304), 512)
        self.fc5 = torch.nn.Linear((512), 2)
        torch.nn.init.kaiming_normal_(self.fc4.weight)
        torch.nn.init.kaiming_normal_(self.fc5.weight)
        self.actor_layer = torch.nn.LogSoftmax(dim=-1)
        
        # critic 
        #self.fc6 = torch.nn.Linear((512), 1)
        #torch.nn.init.kaiming_normal_(self.fc6.weight)

    def forward(self, x):
        c1 = F.relu(self.conv1(x))
        c2 = F.relu(self.conv2(c1))
        c3 = F.relu(self.conv3(c2))
        f4 = F.relu(self.fc4(flatten(c3)))
        #critic_value = self.fc6(f4)
        policy_value = self.actor_layer(self.fc5(f4))

        return policy_value #, critic_value


    

class PPO():
    def __init__(self, device, state_dim, action_dim, n_latent_var, lr, gamma, K_epochs, eps_clip):
        self.device = device
        self.lr = lr
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        print("PPO setting...")
        print("learning rate",self.lr)
        print("gamma",self.gamma)
        print("eps_clip",self.eps_clip)
        print("K_epochs", self.K_epochs)
        self.policy = ActorCritic(state_dim, action_dim, n_latent_var).to(device)
        self.policy_old = copy.deepcopy(self.policy).to(device)
        self.policy_old.eval()
        self.optimizer = torch.optim.RMSprop(self.policy.parameters(),lr=lr)

        self.MseLoss = torch.nn.MSELoss()


    def act(self, state, memory, tensorboard): # action choosing
        state_expand = torch.from_numpy(np.expand_dims(state,0)).float().to(self.device) 
        with torch.no_grad():
            #logits, value = self.policy_old(state_expand)
            logits = self.policy_old(state_expand)
            action_probs = torch.exp(logits)
            
            # ----- debug for implementation error ------ 
            #tensorboard.histogram_summary("logits",logits)
            #tensorboard.histogram_summary("actions probs",action_probs)

            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()
            
            memory.states.append(torch.from_numpy(state).float())
            memory.actions.append(action)
            memory.logprobs.append(dist.log_prob(action))
            return action.item(), value.item()
    
    def evaluate(self, state, action): # prepare for PPO Loss
        #logits, values = self.policy(state)
        logits = self.policy(state)
        action_probs = torch.exp(logits)
        dist = torch.distributions.Categorical(action_probs) 
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        # sanity check
        #p_log_p = logits * action_probs
        #sanity_check_cross_entropy = -p_log_p.sum(-1)

        # return action_logprobs, torch.squeeze(values), dist_entropy
        return action_logprobs, torch.zeros((1)), dist_entropy
    def montecarlo_discounted_rewards(self, memory):
        # Monte Carlo estimate of state rewards:
        rewards = []
        discounted_reward = 0
        for reward in reversed(memory.rewards):
            if reward != 0:
                discounted_reward = 0 # pong specific ! (game boundary)
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        
        # Normalizing the rewards:
        rewards = torch.tensor(rewards).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        return rewards
    
    def gae(self, memory, gamma=0.99, lambd=1.0):
        # reference code : https://github.com/JasonYao81000/MLDS2018SPRING/blob/master/hw4/agent_dir/agent_pg.py
        v_preds = memory.values
        v_preds_next = v_preds[1:] + [0]
        deltas = [r_t + gamma * v_next - v for r_t, v_next, v in zip(memory.rewards, v_preds_next, v_preds)]
        # calculate generative advantage estimator(lambda = 1), see ppo paper eq(11)
        gaes = copy.deepcopy(deltas)
        for t in reversed(range(len(gaes) - 1)):  # is T-1, where T is time step which run policy
            gaes[t] = gaes[t] + gamma * gaes[t + 1]
        
        return gaes

    def update(self, memory, tensorboard):   
        self.policy.train()
        tensorboard.time_s += 1

        #rewards = self.montecarlo_discounted_rewards(memory)
        #advantages = self.gae(memory)
        dis_rewards = self.montecarlo_discounted_rewards(memory)
        advantages = dis_rewards
        # convert list to tensor
        old_states = torch.stack(memory.states).to(self.device).detach()  
        old_actions = torch.stack(memory.actions).to(self.device).detach()
        old_logprobs = torch.stack(memory.logprobs).to(self.device).detach()


        old_values = torch.FloatTensor(memory.values).to(self.device).detach()
        old_rewards = torch.FloatTensor(memory.rewards).to(self.device).detach()
        #advantages = torch.FloatTensor(advantages).to(self.device).detach()
        # normalization
        #advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)
        # target values
        returns = old_values + advantages

        #tensorboard.histogram_summary("old_actions", old_actions)
        #tensorboard.histogram_summary("old_rewards", old_rewards)        
        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # Evaluating old actions and values :
            logprobs, state_values, dist_entropy = self.evaluate(old_states, old_actions)
            #tensorboard.histogram_summary("probs", torch.exp(logprobs))
            #tensorboard.histogram_summary("state_values", state_values)
            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs.detach())
                
            # Finding Surrogate Loss:
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            #mse_loss = 0.5*self.MseLoss(state_values, returns)
            #cross_entropy_loss = 0.01 * dist_entropy
            loss = -torch.min(surr1, surr2) #+ mse_loss - cross_entropy_loss
            #tensorboard.scalar_summary("clamp_loss", (-torch.min(surr1, surr2).mean().item()))
            #tensorboard.scalar_summary("mse_loss", mse_loss.mean().item())
            #tensorboard.scalar_summary("cross_entropy", cross_entropy_loss.mean().item())
            #tensorboard.scalar_summary("total_loss", loss.mean().item())
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            #tensorboard.histogram_summary("now_conv1_weight", self.policy.conv1.weight)
            #tensorboard.histogram_summary("now_conv2_weight", self.policy.conv2.weight)
            #tensorboard.histogram_summary("now_conv3_weight", self.policy.conv3.weight)
            #tensorboard.histogram_summary("now_fc4_weight", self.policy.fc4.weight)
            #tensorboard.histogram_summary("now_fc5_weight", self.policy.fc5.weight)
            #tensorboard.histogram_summary("now_fc6_weight", self.policy.fc6.weight)

            #tensorboard.histogram_summary("old_conv1_weight", self.policy_old.conv1.weight)
            #tensorboard.histogram_summary("old_conv2_weight", self.policy_old.conv2.weight)
            #tensorboard.histogram_summary("old_conv3_weight", self.policy_old.conv3.weight)
            #tensorboard.histogram_summary("old_fc4_weight", self.policy_old.fc4.weight)
            #tensorboard.histogram_summary("old_fc5_weight", self.policy_old.fc5.weight)
            #tensorboard.histogram_summary("old_fc6_weight", self.policy_old.fc6.weight)

        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())
        
def prepro(I,image_size=[80,80]):
    """
    Call this function to preprocess RGB image to grayscale image if necessary
    This preprocessing code is from
        https://github.com/hiwonjoon/tf-a3c-gpu/blob/master/async_agent.py 
    Input: 
    RGB image: np.array
        RGB screen of game, shape: (210, 160, 3)
    Default return: np.array 
        Grayscale image, shape: (80, 80, 1)
    
    """
    """
    # obsv : [210, 180, 3] HWC
    # preprocessing code is partially adopted from https://github.com/carpedm20/deep-rl-tensorflow
    y = 0.2126 * o[:, :, 0] + 0.7152 * o[:, :, 1] + 0.0722 * o[:, :, 2]
    y = y.astype(np.uint8)
    #Scipy actually requires WHC images, but it doesn't matter.
    resized = scipy.misc.imresize(y, image_size)
    return np.expand_dims(resized.astype(np.float32),axis=2)
    """    

    #Karpathy method(I think this preprocessing is better than above)
    I = I[35:195] # crop
    I = I[::2,::2,0] # downsample by factor of 2
    I[I == 144] = 0 # erase background (background type 1)
    I[I == 109] = 0 # erase background (background type 2)
    I[I != 0] = 1 # everything else (paddles, ball) just set to 1


    return (np.expand_dims(I.astype(np.float),2)).transpose((2, 0, 1))
    


class Agent_PG(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """

        super(Agent_PG,self).__init__(env)
        if args.test_pg:
            #you can load your model here
            print('loading trained model')

        ##################
        # YOUR CODE HERE #
        self.init_action = None
        print("Setting Environment ...")
        self.env = env
        self.env.seed(seed)

        print("Environment info ...")
        print("  action space",self.env.get_action_space())
        print("  meaning:", self.env.get_meaning())
        print("  obeservation space",self.env.get_observation_space())
        
        print("-- Debug -- ")
        print("def discount rewards: in debug_agent_pg.py : True")
        
        
        # Model : Neural Network
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else :
            self.device = torch.device('cpu')
        print("Device...  ",self.device)
        self.net = PPO(self.device, 80*80, 2, 256, lr=1e-3, gamma=0.99, K_epochs=4, eps_clip=0.2)
        self.memory = Memory()
        self.tensorboard = TensorboardLogger("./logger_ppo/exp2_7_20")



        ##################


    def init_game_setting(self):
        """

        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary

        """
        pass


    def train(self):
        """
        Implement your training algorithm here
        """
        ##################
        # YOUR CODE HERE #
        # Test shape
        #s = self.env.reset()
        #print("s shape", s.shape)
        #o = prepro(s)
        #o = o.reshape(80,80)
        #print("o shape", o.shape)
        #print(o)
        #plt.imshow(o,cmap='gray',vmin=0,vmax=255)
        #plt.show()

        #print("o shape", o.shape)
        #sample_action = self.make_action(o, test=True)
        #print("sample action", sample_action)
        #print(pe.predict(s))
        #print(pe.network(torch.FloatTensor(s)))
        #plt.plot(range(10), range(10))
        #plt.savefig('f.png')
        
        NN = 1000
        MAX_GAME_FRAME = 19000
        episode_reward = np.array([])
        loss_history = np.array([])
        total_rewards = []

        for episode in range(NN):
            # Collect Data (s,a,r)
            s_0 = prepro(self.env.reset()) # reset environment
            sample_action = self.env.action_space.sample()
            s_1, _, _, _ = self.env.step(sample_action)
            s_1 = prepro(s_1)
            for _ in range(MAX_GAME_FRAME): 
                delta_state = s_1 - s_0
                s_0 = s_1              

                action = self.make_action(delta_state) # logprob is for PPO
                s_1, reward, done, info = self.env.step(action)
                s_1 = prepro(s_1)
                
                # Store reward, old_policy_value
                self.memory.rewards.append(reward)
                self.memory.values.append(value)
                if done:
                    break

            total_reward = np.sum(self.memory.rewards)
            total_rewards.append(total_reward)
            episode_reward = np.append(episode_reward, total_reward)


            # update
            self.net.update(self.memory, self.tensorboard)
            
            # clear
            self.memory.clear_memory()
            
            # record
            print("\rEp: {} Average of last 10: {:.2f}".format(
                episode + 1, np.mean(total_rewards[-30:])), end="")    
            self.tensorboard.scalar_summary("average_reward", np.mean(total_rewards[-30:]))
        plt.plot(range(NN),episode_reward)
        plt.savefig('ppo_loss.png')
        ##################
 

    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent

        Input:
            observation: np.array
                current RGB screen of game, shape: (210, 160, 3)

        Return:
            action: int
                the predicted action from trained model
        """
        ##################
        # YOUR CODE HERE #
        ##################
        # Feedforward of the Network

        action = self.net.act(observation, self.memory, self.tensorboard)
        return int(action + 2)
        

    def discount_reward(self, reward_tensor, t, Tn, discount_factor=0.99):
        sum = 0.0
        for t_prime in range(t,Tn): # range(t,Tn+1) = [t, t+1, ..., Tn]
            sum += (np.power(discount_factor, t_prime-t) * reward_tensor[t_prime]) # factor * reward
        return sum
    def karparthy_discount_reward(self, r):
        """ take 1D float array of rewards and compute discounted reward """
        gamma = 0.99
        discounted_r = np.zeros_like(r)
        running_add = 0
        for t in reversed(range(0, r.shape[0])):
            if r[t] != 0 : 
                running_add = 0 # reset the sum, since this was a game boundary (pong specific!)
            running_add = running_add * gamma + r[t]
            discounted_r[t] = running_add
        return discounted_r

    def flatten(self,x):
        N = x.shape[0] # read in N, C, H, W
        return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image


    def discount_rewards(self,rewards, gamma=0.99):
        r = np.array([gamma**i * rewards[i] 
                    for i in range(len(rewards))])
        # Reverse the array direction for cumsum and then
        # revert back to the original order
        r = r[::-1].cumsum()[::-1]
        return r - r.mean()
    
    def debugger(self, observation, count, action=5):
        '''
        Input :
            count : means each frame of the postfix image file name 
            action : test the action meaning

        Output :
            debug_image/ : directory of one epoch images
            debug_action.txt : action text file
        '''
        cv2.imwrite("debug_image/"'output'+str(count)+str('.jpg'), observation)
        fp = open("debug_action.txt", "a")
        fp.writelines(str(action))
        fp.close()
