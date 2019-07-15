import numpy as np

def debug_discount_rewards():
    '''
    t1 = np.ones((10))
    t2 = np.ones((10))
    for t in range(10):
        t1[t] = self.discount_reward(t1, t, 10)
    print("t1 is:", t1)
    print("t1 is:", self.discount_rewards(t2))
        
    return True
    '''

def discount_rewards(r, gamma=0.99):
    """ take 1D float array of rewards and compute discounted reward
    Source: https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5)
    """

    eps = 1e-10
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, len(r))):
        if r[t] != 0:
            running_add = 0  # reset the sum, since this was a game boundary (pong specific!)
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    
    print("discounted_r: ", discounted_r)
    # standardize the rewards to be unit normal (helps control the gradient estimator variance)
    discounted_r -= np.mean(discounted_r)
    discounted_r /= np.std(discounted_r) + eps
    #import matplotlib.pyplot as plt; plt.plot(discounted_r); plt.show()
    return discounted_r


if __name__ == '__main__':
    print(3)
    r = np.ones((5))
    r[0] = 0
    r[1] = 0
    r[2] = 0
    r[3] = -1
    print(discount_rewards(r))