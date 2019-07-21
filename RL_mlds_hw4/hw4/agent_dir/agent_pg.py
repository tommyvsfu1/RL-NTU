from agent_dir.agent import Agent
import scipy
import scipy.misc
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import skimage

seed = 9487
np.random.seed(seed)
torch.manual_seed(seed)

class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.action_prbo = []
    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.action_prbo[:]

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
    

class PPO:
    def __init__(self):
        self.lr = 1e-3
        self.betas = 1
        self.gamma = 0.99
        self.eps_clip = 0.2
        self.K_epochs = 1
        self.device = 'cpu'
        D_in, H, D_out = 80*80, 256, 2
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
    return I.astype(np.float).ravel()
    


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
        self.device = torch.device('cpu')
        print("Device...  ",self.device)

        self.improvement = "PG"
        self.net = PPO()
        self.memory = Memory()
        self.optimizer = torch.optim.RMSprop(self.net.policy.parameters(), lr=1e-3)
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
        episode_reward = np.array([])
        loss_history = np.array([])
        total_rewards = []
        for episode in range(NN):
            # Collect Data (s,a,r)        
            s_0 = self.env.reset() # reset environment
            s_0 = prepro(s_0)
            sample_action = self.env.action_space.sample()
            s_1, _, _, _ = self.env.step(sample_action)
            s_1 = prepro(s_1)
            for _ in range(MAX_GAME_FRAME): 
                delta_state = s_1 - s_0
                s_0 = s_1              

                action, action_prob = self.make_action(delta_state) # logprob is for PPO
                s_1, reward, done, info = self.env.step(action)
                s_1 = prepro(s_1)
                
                # Store state
                self.memory.states.append(torch.from_numpy(delta_state))
                self.memory.actions.append(torch.tensor([action-2]))
                self.memory.rewards.append(reward)
                self.memory.action_prbo.append(action_prob)
                self.memory.logprobs.append(torch.log( torch.tensor([action_prob[action-2]]) ))
                if done:
                    break
            
            total_reward = np.sum(self.memory.rewards)
            total_rewards.append(total_reward)
            # Discount and Normalize rewards
            reward_tensor = self.karparthy_discount_reward(self.memory.rewards)
            advantage_function = (reward_tensor - np.mean(reward_tensor)) / (np.std(reward_tensor) + 1e-5)
            advantage_function =  (torch.from_numpy(advantage_function)).float()
            # update
            if self.improvement == "PG":
                self.vanilla_update(advantage_function)
            # record        
            print("episode",episode,"last action",self.memory.actions[-1],"action prob",self.memory.action_prbo[-1],"average reward", np.mean(total_rewards[-30:]))
            # clear
            self.memory.clear_memory()
        plt.plot(range(NN),episode_reward)
        plt.savefig('pg_loss.png')
        ##################

    def vanilla_update(self,advantage_function):
        # gradient  (reference : p.29 http://rll.berkeley.edu/deeprlcourse/f17docs/lecture_4_policy_gradient.pdf)
        self.net.policy.train()        
        x = torch.stack(self.memory.states).float().to(self.device).detach()
        action_tensor = torch.stack(self.memory.actions).to(self.device).detach()
        advantage_function = advantage_function.to(self.device)
        old_action_probs = torch.stack(self.memory.logprobs).reshape(-1).to(self.device).detach()
        flatten_x = self.flatten(x).to(self.device) 
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
        for _ in range(4):
            logits = self.net.policy(flatten_x)  
            new_action_prob = logits.gather(1, action_tensor).reshape(-1) #(N)
            r = torch.exp((new_action_prob - old_action_probs))
            loss1 = r * advantage_function
            loss2 = torch.clamp(r, 1-0.2, 1+0.2) * advantage_function
            loss = -torch.min(loss1, loss2)
            loss = torch.mean(loss)
        
                        
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
                flatten_x = self.flatten(x)
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
                x = x.float() # type conversion
                flatten_x = self.flatten(x)
                flatten_x = flatten_x.to(self.device)
                log_logits = self.net.policy_old(flatten_x)
                action_prob = np.exp(log_logits.cpu().numpy()[0]) 
                action = np.random.choice(range(2), p=action_prob)
                return int(action + 2), action_prob
        

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
