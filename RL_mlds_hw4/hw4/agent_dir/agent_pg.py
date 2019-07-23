from agent_dir.agent import Agent
import scipy
import scipy.misc
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import skimage
from logger import TensorboardLogger
import torch.nn.functional as F  # useful stateless functions
import copy
seed = 11037
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.action_prob = []
        self.values = []
    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.action_prob[:]
        del self.values[:]
def flatten(x):
    N = x.shape[0] # read in N, C, H, W
    return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image

class ActorCritic(torch.nn.Module):
    def __init__(self, state_dim, action_dim, n_latent_var):
        super(ActorCritic, self).__init__()
        self.affine = torch.nn.Linear(state_dim, n_latent_var)
        
        # actor
        self.action_layer = torch.nn.Sequential(
                torch.nn.Linear(state_dim, n_latent_var),
                torch.nn.Tanh(),
                torch.nn.Linear(n_latent_var, n_latent_var),
                torch.nn.Tanh(),
                torch.nn.Linear(n_latent_var, action_dim),
                torch.nn.Softmax(dim=-1)
                )
        
        # critic
        self.value_layer = torch.nn.Sequential(
                torch.nn.Linear(state_dim, n_latent_var),
                torch.nn.Tanh(),
                torch.nn.Linear(n_latent_var, n_latent_var),
                torch.nn.Tanh(),
                torch.nn.Linear(n_latent_var, 1)
                )
        
    def forward(self):
        raise NotImplementedError
        
    def act(self, state, memory): # PI
        state = torch.from_numpy(state).float().to(device) 
        action_probs = self.action_layer(state)
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        
        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(dist.log_prob(action))
        
        return action.item()
    
    def evaluate(self, state, action): # V(s)
        action_probs = self.action_layer(state)
        dist = torch.distributions.Categorical(action_probs)
        
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        
        state_value = self.value_layer(state)
        
        return action_logprobs, torch.squeeze(state_value), dist_entropy
    

class MLP(torch.nn.Module):
    def __init__(self):
        super(MLP, self).__init__()  
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
        self.fc6 = torch.nn.Linear((512), 1)
        torch.nn.init.kaiming_normal_(self.fc6.weight)

    def forward(self, x):
        c1 = F.relu(self.conv1(x))
        c2 = F.relu(self.conv2(c1))
        c3 = F.relu(self.conv3(c2))
        f4 = F.relu(self.fc4(flatten(c3)))
        critic_value = self.fc6(f4)
        policy_value = self.actor_layer(self.fc5(f4))

        return policy_value, critic_value 

class PPO():
    def __init__(self,device):
        self.lr = 1e-4
        self.betas = 1
        self.gamma = 0.99
        self.eps_clip = 0.2
        self.K_epochs = 1
        self.device = device
        print("PPO using device:", self.device)
        D_in, H, D_out = 80*80, 256, 2
        """
        self.policy = torch.nn.Sequential(
            torch.nn.Linear(D_in, H),
            torch.nn.ReLU(),
            torch.nn.Linear(H, D_out),
            torch.nn.LogSoftmax(dim=-1)
        ).to(self.device)
        self.policy_old = torch.nn.Sequential(
            torch.nn.Linear(D_in, H),
            torch.nn.ReLU(),
            torch.nn.Linear(H, D_out),
            torch.nn.LogSoftmax(dim=-1)
        ).to(self.device)
        """
        self.policy = MLP().to(self.device)
        self.policy_old = MLP().to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.optimizer = torch.optim.Adam(self.policy.parameters(),
                                              lr=self.lr)

        
        self.MseLoss = torch.nn.MSELoss()
    
    def update(self, memory):   
        # Monte Carlo estimate of state rewards:
        rewards = []
        discounted_reward = 0
        for reward in reversed(memory.rewards):
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        
        # Normalizing the rewards:
        rewards = torch.tensor(rewards).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        
        # convert list to tensor
        old_states = torch.stack(memory.states).to(device).detach()
        old_actions = torch.stack(memory.actions).to(device).detach()
        old_logprobs = torch.stack(memory.logprobs).to(device).detach()
        
        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # Evaluating old actions and values :
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            
            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs.detach())
                
            # Finding Surrogate Loss:
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - 0.01*dist_entropy
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
        
        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())
        
    def optimize_model(self):
        pass
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
    #return I.astype(np.float).ravel().transpose((2, 0, 1)) # 2 layer FC
    return (np.expand_dims(I.astype(np.float),2)).transpose((2, 0, 1))  # MLP
    
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
        self.tensorboard = TensorboardLogger("./PPO_v2/exp2_batch/")
        self.improvement = "PPO"
        self.net = PPO(self.device)
        self.memory = Memory()
        self.optimizer = torch.optim.RMSprop(self.net.policy.parameters(), lr=1e-4)
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
        NN = 1000
        MAX_GAME_FRAME = 19000
        NUM_WORKER = NUM_EPISODE = 1
        episode_reward = np.array([])
        loss_history = np.array([])
        total_rewards = []
        reward_sum_running_avg = 0
        for iteraition in range(NN):
            reward_history = []
            for w in range(NUM_WORKER):
                # Collect Data (s,a,r)        
                s_0 = self.env.reset() # reset environment
                s_0 = prepro(s_0)
                sample_action = self.env.action_space.sample()
                s_1, _, _, _ = self.env.step(sample_action)
                s_1 = prepro(s_1)
                for t in range(MAX_GAME_FRAME): 
                    delta_state = s_1 - s_0
                    s_0 = s_1              

                    action, action_prob, v = self.make_action(delta_state) # logprob is for PPO
                    s_1, reward, done, info = self.env.step(action)
                    s_1 = prepro(s_1)
                    
                    # Store state
                    self.memory.states.append(torch.from_numpy(delta_state))
                    self.memory.actions.append(torch.tensor([action-2]))
                    self.memory.rewards.append(reward)
                    self.memory.action_prob.append(action_prob)
                    self.memory.logprobs.append(torch.log( torch.tensor([action_prob[action-2]]) ))
                    self.memory.values.append(v.item())
                    reward_history.append(reward)
                    if done:
                        reward_sum = sum(reward_history[-t:])
                        reward_sum_running_avg = 0.99*reward_sum_running_avg + 0.01*reward_sum if reward_sum_running_avg else reward_sum
                        #print('Iteration %d, Episode %d  - last_action: %d, last_action_prob: %.2f, reward_sum: %.2f, running_avg: %.2f' % (iteraition, w, self.memory.actions[-1],self.memory.action_prob[-1], reward_sum, reward_sum_running_avg))
                        print("Iteration",iteraition,"pisode",w,"last_action",self.memory.actions[-1],"action_prob",self.memory.action_prob[-1],"reward_sum",reward_sum,"running_average",reward_sum_running_avg)
                        break
            
            total_reward = np.sum(self.memory.rewards)
            total_rewards.append(total_reward)
            # Discount and Normalize rewards
            #reward_tensor = self.karparthy_discount_reward(self.memory.rewards)
            reward_tensor = self.gae()
            advantage_function = (reward_tensor - np.mean(reward_tensor)) / (np.std(reward_tensor) + 1e-5)
            advantage_function =  (torch.from_numpy(advantage_function)).float()
            # update
            if self.improvement == "PPO":
                self.vanilla_update(advantage_function)
            # record        
            print("iteration",iteraition,"last action",self.memory.actions[-1],"action prob",self.memory.action_prob[-1],"average reward", np.mean(total_rewards[-30:]))
            self.tensorboard.scalar_summary("average_reward",np.mean(total_rewards[-30:]))
            # clear
            self.memory.clear_memory()
        plt.plot(range(NN),episode_reward)
        plt.savefig('pg_loss.png')
        ##################

    def gae(self,gamma=0.99, lambd=1.0):
        # reference code : https://github.com/JasonYao81000/MLDS2018SPRING/blob/master/hw4/agent_dir/agent_pg.py
        v_preds = self.memory.values
        v_preds_next = v_preds[1:] + [0]
        deltas = [r_t + gamma * v_next - v for r_t, v_next, v in zip(self.memory.rewards, v_preds_next, v_preds)]
        # calculate generative advantage estimator(lambda = 1), see ppo paper eq(11)
        gaes = copy.deepcopy(deltas)
        for t in reversed(range(len(gaes) - 1)):  # is T-1, where T is time step which run policy
            gaes[t] = gaes[t] + gamma * gaes[t + 1]
        
        return gaes

    def vanilla_update(self,advantage_function):
        # gradient  (reference : p.29 http://rll.berkeley.edu/deeprlcourse/f17docs/lecture_4_policy_gradient.pdf)
        self.net.policy.train()        
        x = torch.stack(self.memory.states).float().to(self.device).detach()
        action_tensor = torch.stack(self.memory.actions).to(self.device).detach()
        advantage_function = advantage_function.to(self.device)
        old_action_probs = torch.stack(self.memory.logprobs).reshape(-1).to(self.device).detach()
        #flatten_x = flatten(x).to(self.device) 
        flatten_x = x.to(self.device)
        #if no softmax
        #negative_log_likelihoods_fn = torch.nn.CrossEntropyLoss(reduction='none') # do not divide by batch, and return vector
        #negative_log_likelihoods = negative_log_likelihoods_fn(logits, action_tensor - 1) # loss = (Tn,)
        #print("negative log likelihoods", negative_log_likelihoods)
        #loss = ( torch.dot(negative_log_likelihoods, advantage_function) ).sum() / N
        
        #else
        #logprob = torch.log(logits)

        """ PG loss
        selected_logprobs = advantage_function * \
                        logits[np.arange(len(action_tensor)), action_tensor]
        loss = (-selected_logprobs.mean())           
        self.optimizer.zero_grad()
        loss.backward()
        """

        """ PPO loss """
        #vs = np.array([[1., 0.], [0., 1.]])
        #ts = torch.FloatTensor(vs[action.cpu().numpy()])
        reward_ = torch.from_numpy(np.array(self.memory.rewards)).float().to(self.device)
        v_preds_next = self.memory.values[1:] + [0]
        pred_next = torch.from_numpy(np.array(v_preds_next)).float().to(self.device)
        fac_pred_next = torch.tensor([0.99]) * pred_next
        for _ in range(4):
            self.tensorboard.time_s += 1
            logits, v = self.net.policy(flatten_x)  
            new_action_prob = logits.gather(1, action_tensor).reshape(-1) #(N)
            
            # clamp loss
            r = torch.exp((new_action_prob - old_action_probs))
            loss1 = r * advantage_function
            loss2 = torch.clamp(r, 1-0.2, 1+0.2) * advantage_function
            
            # mse
            mse_loss = 0.5*self.net.MseLoss(v.reshape(-1), reward_+fac_pred_next)
            
            # entropy
            action_probs = torch.exp(logits)
            dist = torch.distributions.Categorical(action_probs) 
            entropy = 0.01*dist.entropy()

            # mean
            loss = -torch.min(loss1, loss2) + mse_loss - entropy 
            loss = torch.mean(loss)

            self.tensorboard.scalar_summary("loss", loss)
                        
            # Update theta 
            # Note : 
            # argmin -log(likelihood) = argmax log(likelihood)
            # that is, gradient descent of -log(likelihood) is equivalent to gradient ascent of log(likelihood)
            # we can call pytorch step() function, just like usual deep learning problem !
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
   
        self.net.policy_old.load_state_dict(self.net.policy.state_dict())


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
        
        if self.improvement == "PG":
            self.net.policy.eval()
            with torch.no_grad():
                x = np.expand_dims(observation, axis=0) # convert to (1,x.shape)
                x = torch.from_numpy(x) # numpy to tensor
                x = x.float() # type conversion
                flatten_x = flatten(x)
                flatten_x = flatten_x.to(self.device)
                log_logits = self.net.policy(flatten_x)
                action_prob = np.exp(log_logits.cpu().numpy()[0]) 
                action = np.random.choice(range(2), p=action_prob)
                return int(action + 2), action_prob
        elif self.improvement == "PPO":
            self.net.policy_old.eval()
            with torch.no_grad():
                x = np.expand_dims(observation, axis=0) # convert to (1,x.shape)
                x = torch.from_numpy(x) # numpy to tensor
                x = x.float().to(self.device) # type conversion
                """ FC
                #flatten_x = flatten(x)
                #flatten_x = flatten_x.to(self.device)
                """
                log_logits, v = self.net.policy_old(x)
                action_prob = np.exp(log_logits.cpu().numpy()[0]) 
                action = np.random.choice(range(2), p=action_prob)
                return int(action + 2), action_prob, v
        

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
        for t in reversed(range(0, len(r))):
            if r[t] != 0 : 
                running_add = 0 # reset the sum, since this was a game boundary (pong specific!)
            running_add = running_add * gamma + r[t]
            discounted_r[t] = running_add
        return discounted_r




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
