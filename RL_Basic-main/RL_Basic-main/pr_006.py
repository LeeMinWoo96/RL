import gym
import numpy as np
import matplotlib.pyplot as plt
from gym.envs.registration import register


policy = {0: 1, 1: 2, 2: 1, 3: 0, 4: 1, 6: 1,
          8: 2, 9: 1, 10: 1, 13: 2, 14: 2}

register(id='FrozenLakeNoSlippery-v0',
         entry_point="gym.envs.toy_text:FrozenLakeEnv",
         kwargs={'map_name': '4x4', 'is_slippery': False},
         )

env = gym.make('FrozenLakeNoSlippery-v0')
env = gym.make('FrozenLake-v0')

n_games = 1000
win_pct = []
scores = []

for i in range(n_games):
    done = False 
    obs = env.reset()
    score = 0

    while not done:
        action = policy[obs] # env 환경이 랜덤하게 Action 을 정해줌
        obs, reward ,done ,info = env.step(action) # info 는 debug 목적 
        score += reward 
        # env.render() # view
    
    scores.append(score)
    if i % 10 == 0:
        average = np.mean(scores[-10:])
        win_pct.append(average)

plt.plot(win_pct)
plt.xlabel('episode')
plt.ylabel('success ratio')
plt.show()

