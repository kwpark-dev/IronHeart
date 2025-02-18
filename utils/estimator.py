import numpy as np



def monte_carlo_return(reward, gamma=0.99):
    reward = np.array(reward)
    
    w = np.arange(len(reward))
    w = w - w[:, np.newaxis]
    w = np.triu(gamma ** w.clip(min=0)).T
    # print(w)
    
    mc_value = (reward.reshape(-1, 1) * w).sum(axis=0)
    
    return mc_value




if __name__ == "__main__":
    reward = [1 ,1 ,1,1,1,1]
    
    val = monte_carlo_return(reward)
    print(val)