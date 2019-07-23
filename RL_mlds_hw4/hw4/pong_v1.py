

import random
import gym
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from torch import nn
from logger import TensorboardLogger


tensorboard = TensorboardLogger("./logger_ppo/debug_721/v2/")
class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()

        self.gamma = 0.99
        self.eps_clip = 0.1

        self.layers = nn.Sequential(
            nn.Linear(6000, 512), nn.ReLU(),
            nn.Linear(512, 2),
        )

    def state_to_tensor(self, I):
        """ prepro 210x160x3 uint8 frame into 6000 (75x80) 1D float vector. See Karpathy's post: http://karpathy.github.io/2016/05/31/rl/ """
        if I is None:
            return torch.zeros(1, 6000)
        I = I[35:185] # crop - remove 35px from start & 25px from end of image in x, to reduce redundant parts of image (i.e. after ball passes paddle)
        I = I[::2,::2,0] # downsample by factor of 2.
        I[I == 144] = 0 # erase background (background type 1)
        I[I == 109] = 0 # erase background (background type 2)
        I[I != 0] = 1 # everything else (paddles, ball) just set to 1. this makes the image grayscale effectively
        return torch.from_numpy(I.astype(np.float32).ravel()).unsqueeze(0)

    def pre_process(self, x, prev_x):
        return self.state_to_tensor(x) - self.state_to_tensor(prev_x)

    def convert_action(self, action):
        return action + 2

    def forward(self, d_obs, action=None, action_prob=None, advantage=None, deterministic=False):
        if action is None:
            with torch.no_grad():
                logits = self.layers(d_obs)
                if deterministic:
                    action = int(torch.argmax(logits[0]).detach().cpu().numpy())
                    action_prob = 1.0
                else:
                    c = torch.distributions.Categorical(logits=logits)
                    action = int(c.sample().cpu().numpy()[0])
                    action_prob = float(c.probs[0, action].detach().cpu().numpy())
                return action, action_prob
        '''
        # policy gradient (REINFORCE)
        logits = self.layers(d_obs)
        loss = F.cross_entropy(logits, action, reduction='none') * advantage
        return loss.mean()
        '''

        # PPO
        vs = np.array([[1., 0.], [0., 1.]])
        ts = torch.FloatTensor(vs[action.cpu().numpy()])

        logits = self.layers(d_obs)

        print("mul", (F.softmax(logits, dim=1) * ts).shape)
        print("action_prob shape", action_prob.shape)
        r = torch.sum(F.softmax(logits, dim=1) * ts, dim=1) / action_prob
        tensorboard.histogram_summary("prob", torch.sum(F.softmax(logits, dim=1) * ts, dim=1) )
        tensorboard.histogram_summary("old_prob", action_prob)
        tensorboard.histogram_summary("ratio", r)
        tensorboard.histogram_summary("advantage", advantage)
        print("r shape", r.shape)
        loss1 = r * advantage
        print("advantage", advantage.shape)
        loss2 = torch.clamp(r, 1-self.eps_clip, 1+self.eps_clip) * advantage
        print("loss1 shape", loss1.shape)
        print("loss2 shape", loss2.shape)
        loss = -torch.min(loss1, loss2)
        print("loss shape", loss.shape)
        loss = torch.mean(loss)
        """
        mul torch.Size([24576, 2])
        action_prob shape torch.Size([24576])
        old_action_prob shape torch.Size([24576])
        r shape torch.Size([24576])
        advantage torch.Size([24576])
        loss1 shape torch.Size([24576])
        loss2 shape torch.Size([24576])
        """
        return loss

env = gym.make('PongNoFrameskip-v4')
env.reset()

policy = Policy()

opt = torch.optim.Adam(policy.parameters(), lr=1e-3)

reward_sum_running_avg = None
for it in range(100000):
    d_obs_history, action_history, action_prob_history, reward_history = [], [], [], []
    for ep in range(10):
        obs, prev_obs = env.reset(), None
        for t in range(190000):
            #env.render()

            d_obs = policy.pre_process(obs, prev_obs)
            with torch.no_grad():
                action, action_prob = policy(d_obs)

            prev_obs = obs
            obs, reward, done, info = env.step(policy.convert_action(action))

            d_obs_history.append(d_obs)
            action_history.append(action)
            action_prob_history.append(action_prob)
            reward_history.append(reward)

            if done:
                reward_sum = sum(reward_history[-t:])
                reward_sum_running_avg = 0.99*reward_sum_running_avg + 0.01*reward_sum if reward_sum_running_avg else reward_sum
                print('Iteration %d, Episode %d (%d timesteps) - last_action: %d, last_action_prob: %.2f, reward_sum: %.2f, running_avg: %.2f' % (it, ep, t, action, action_prob, reward_sum, reward_sum_running_avg))
                break

    # compute advantage
    R = 0
    discounted_rewards = []

    for r in reward_history[::-1]:
        if r != 0: R = 0 # scored/lost a point in pong, so reset reward sum
        R = r + policy.gamma * R
        discounted_rewards.insert(0, R)

    discounted_rewards = torch.FloatTensor(discounted_rewards)
    discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / discounted_rewards.std()

    # update policy
    print("length of action history", len(action_history))
    print("5 * 24576", 5*24576)
    for _ in range(5):
        n_batch = 24576
        idxs = random.sample(range(len(action_history)), n_batch)
        d_obs_batch = torch.cat([d_obs_history[idx] for idx in idxs], 0)
        action_batch = torch.LongTensor([action_history[idx] for idx in idxs])
        action_prob_batch = torch.FloatTensor([action_prob_history[idx] for idx in idxs])
        advantage_batch = torch.FloatTensor([discounted_rewards[idx] for idx in idxs])
        tensorboard.time_s += 1
        opt.zero_grad()
        loss = policy(d_obs_batch, action_batch, action_prob_batch, advantage_batch)
        loss.backward()
        opt.step()

    if it % 5 == 0:
        torch.save(policy.state_dict(), 'params.ckpt')

env.close()
