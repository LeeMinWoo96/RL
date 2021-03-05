import gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make('FrozenLake-v0')
n_games = 1000
win_pct = []
scores = []

for i in range(n_games):
    done = False 
    obs = env.reset()
    score = 0

    while not done:
        action = env.action_space.sample() # env 환경이 랜덤하게 Action 을 정해줌
        obs, reward ,done ,info = env.step(action) # info 는 debug 목적 
        score += reward 
        env.render() # view
    
    scores.append(score)
    if i % 10 == 0:
        average = np.mean(scores[-10:])
        win_pct.append(average)

plt.plot(win_pct)
plt.xlabel('episode')
plt.ylabel('success ratio')
plt.show()
