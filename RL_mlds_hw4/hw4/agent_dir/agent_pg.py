from agent_dir.agent import Agent
import scipy
import scipy.misc
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
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
    
    return shape : (6400,)
    """
    '''
    y = o.astype(np.uint8)
    resized = scipy.misc.imresize(y, image_size)
    return np.expand_dims(resized.astype(np.float32),axis=2)
    '''
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
        print("Environment info:")
        print("  action space",self.env.get_action_space())
        print("  obeservation space",self.env.get_observation_space())
        
        print("-- Debug -- ")
        print("def discount rewards: in debug_agent_pg.py : True")
        # Model : Neural Network
        self.device = torch.device('cpu')
        print("using device",self.device)
        D_in, H, D_out = 80*80, 256, 3
        self.model = torch.nn.Sequential(
            torch.nn.Linear(D_in, H),
            torch.nn.ReLU(),
            torch.nn.Linear(H, D_out),
            torch.nn.Softmax(dim=-1)
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
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
        # Test shape
        #s = self.env.reset()
        #o = prepro(s)
        #print("o shape", o.shape)
        #sample_action = self.make_action(o, test=True)
        #print("sample action", sample_action)
        #print(pe.predict(s))
        #print(pe.network(torch.FloatTensor(s)))
        #plt.plot(range(10), range(10))
        #plt.savefig('f.png')




        NN = 1000
        episode_reward = np.array([])
        loss_history = np.array([])
        total_rewards = []
        for episode in range(NN):
            # Collect Data (s,a,r)
            observation_tensor = [] # use list to store image, then convert to numpy
            action_tensor = np.array([]) # since action is integer, use numpy directly to store action
            reward_tensor = np.array([]) # since reward is integer, use numpy directly to store reward
        
            N = 1
            for i_episode in range(N):  # each (episode)
                s_0 = self.env.reset() # reset environment
                s_0 = prepro(s_0)
                s_shape = s_0.shape
                sample_action = self.make_action(s_0)
                while(True):
                    observation_tensor.append(s_0)
                    action = sample_action
                    s_1, reward, done, info = self.env.step(action)
                    # Store state
                    action_tensor = np.append(action_tensor, action)
                    reward_tensor = np.append(reward_tensor, reward)
                    
                    # Sample Action (tau_theta(s))
                    s_0 = prepro(s_1)
                    if s_0.shape != s_shape:
                        print("error", s_0.shape, s_shape)
                    sample_action = self.make_action(s_0)
                    if done:
                        #print("Episode finished after {} timesteps".format(reward_tensor.shape[0]))
                        break
            #print("type:", observation_tensor)
            #observation_tensor = np.array(observation_tensor,dtype=np.float64)
            #print("observation tensor shape", observation_tensor.shape)
            #print("type of observation tensor", type(observation_tensor))
            #observation_tensor = np.expand_dims(observation_tensor, axis=0) 
            
            #print("total reward", np.sum(reward_tensor))
            total_reward = np.sum(reward_tensor)
            episode_reward = np.append(episode_reward, total_reward)
            # Discount and Normalize rewards
            Tn = reward_tensor.shape[0]
            reward_tensor = self.karparthy_discount_reward(reward_tensor)
            #for t in range(Tn):
            #    reward_tensor[t] = self.discount_reward(reward_tensor, t, Tn) 
            #reward_tensor = (reward_tensor - np.mean(reward_tensor)) / (np.std(reward_tensor) + 1e-10) # normalization
            #b = np.sum(reward_tensor) / reward_tensor.shape[0] # expectation of reward
            #advatange_function = (torch.from_numpy(reward_tensor - b)).float()
            advatange_function = (reward_tensor - np.mean(reward_tensor)) / (np.std(reward_tensor) + 1e-10)
            advatange_function =  (torch.from_numpy(advatange_function)).float()
            total_rewards.append(total_reward)


            # action tensor prepro
            action_tensor -= 1


            # gradient  (reference : p.29 http://rll.berkeley.edu/deeprlcourse/f17docs/lecture_4_policy_gradient.pdf)
            self.model.train()
            if type(observation_tensor) == type(list()): # if use list to tensor
                #print("observation shape", len(observation_tensor))
                x = torch.FloatTensor(observation_tensor)
                #print("tensor x shape", x.shape)
            else : # if use nupmy array
                x = (torch.from_numpy(observation_tensor)).float()
            flatten_x = self.flatten(x)
            ### Device: CPU -> GPU
            flatten_x = flatten_x.to(self.device)
            action_tensor = (torch.from_numpy(action_tensor).long()).to(self.device)
            advatange_function = advatange_function.to(self.device)
            ### Compute Loss and gradient
            logits = self.model(flatten_x)
            
            #if no softmax
            #negative_log_likelihoods_fn = torch.nn.CrossEntropyLoss(reduction='none') # do not divide by batch, and return vector
            #negative_log_likelihoods = negative_log_likelihoods_fn(logits, action_tensor - 1) # loss = (Tn,)
            #print("negative log likelihoods", negative_log_likelihoods)
            #loss = ( torch.dot(negative_log_likelihoods, advatange_function) ).sum() / N
            
            #else
            logprob = torch.log(logits)
            selected_logprobs = advatange_function * \
                            logprob[np.arange(len(action_tensor)), action_tensor]
            loss = -selected_logprobs.mean()           
            self.optimizer.zero_grad()
            loss.backward()
            
                        
            # Update theta 
            # Note : 
            # argmin -log(likelihood) = argmax log(likelihood)
            # that is, gradient descent of -log(likelihood) is equivalent to gradient ascent of log(likelihood)
            # we can call pytorch step() function, just like usual deep learning problem !
            self.optimizer.step()

            print("\rEp: {} Average of last 10: {:.2f}".format(
            episode + 1, np.mean(total_rewards[-10:])), end="")
            
            #print(loss)
        plt.plot(range(NN),episode_reward)
        #plt.show()
        plt.savefig('loss.png')
        plt.savefig('loss.pdf')
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
        if test:
            self.model.eval()
            with torch.no_grad():
                #print("observation shape", observation.shape)
                x = np.expand_dims(observation, axis=0) # convert to (1,x.shape)
                x = torch.from_numpy(x) # numpy to tensor
                x = x.float() # type conversion
                #print("x shape", x.shape)
                flatten_x = self.flatten(x)
                #print("flatten_x shape", flatten_x.shape)
                flatten_x = flatten_x.to(self.device)
                outputs = self.model(flatten_x)
                #sample_action = outputs.argmax()
                #print("sample action",sample_action.item())
                sample_action = outputs.cpu().numpy() 
                #print("sample action", sample_action)
                #return sample_action

                action = np.random.choice(range(3), p=sample_action[0])
                return int(action + 1)
        else :
            self.model.train()
            #x = (torch.from_numpy(observation)).float()
            x = observation
            flatten_x = self.flatten(x)
            #print("flatten shape", flatten_x.shape)
            ### Device: CPU -> GPU
            flatten_x = flatten_x.to(self.device)
            ### Compute Loss and gradient
            logits = self.model(flatten_x)
            return logits
        

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

    def reinforce(self, num_episodes=2000,
                batch_size=16, gamma=0.99):
        loss_history = np.array([])
        # Set up lists to hold results
        total_rewards = []
        batch_rewards = []
        batch_actions = []
        batch_states = []
        batch_counter = 1

        #action_space = np.arange(self.env.action_space.n)
        #count = 0
        for ep in range(num_episodes):
            s_0 = prepro(self.env.reset())
            states = []
            rewards = []
            actions = []
            complete = False
            while complete == False:
                #print("++")
                # Get actions and convert to numpy array
                #count += 1
                #self.debugger(s_0, count)
                action = self.make_action(s_0)
                s_1, r, complete, _ = self.env.step(action)
                
                states.append(s_0)
                rewards.append(r)
                actions.append(action)
                s_0 = prepro(s_1)
                
                # If complete, batch data
                if complete:
                    batch_rewards.extend(self.discount_rewards(rewards, gamma))
                    batch_states.extend(states)
                    batch_actions.extend(actions)
                    batch_counter += 1
                    total_rewards.append(sum(rewards))
                    #print("batch action",batch_actions)
                    # If batch is complete, update network
                    if batch_counter == batch_size:
                        batch_rewards = np.array(batch_rewards)
                        batch_states = np.array(batch_states)
                        batch_actions = np.array(batch_actions)
                        self.optimizer.zero_grad()
                        state_tensor = torch.from_numpy((batch_states)).float()
                        reward_tensor = torch.from_numpy((batch_rewards)).float()
                        # Actions are used as indices, must be LongTensor
                        action_tensor = torch.from_numpy((batch_actions)).long() - 1
                        #print("action tensor", action_tensor)
                        # Calculate loss
                        logprob = torch.log(
                            self.make_action(state_tensor,False))
                        #print("logprob shape", logprob.shape)
                        #print("reward_tensor", reward_tensor)
                        #print("logprob", logprob[np.arange(len(action_tensor)), action_tensor])
                        selected_logprobs = reward_tensor * \
                            logprob[np.arange(len(action_tensor)), action_tensor]
                        loss = -selected_logprobs.mean()
                        # Calculate gradients
                        loss.backward()
                        # Apply gradients
                        self.optimizer.step()
                        
                        batch_rewards = []
                        batch_actions = []
                        batch_states = []
                        batch_counter = 1
                        loss_history = np.append(loss_history, loss.item())
                        
                    # Print running average
                    print("\rEp: {} Average of last 10: {:.2f}".format(
                        ep + 1, np.mean(total_rewards[-10:])), end="")

        #plt.plot(range(loss_history.shape[0]), loss_history)
        #plt.show()
        return total_rewards

    
    # custom weights initialization 
    def weights_init(self,m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            m.weight.data.normal_(0.0, 0.02)
    
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