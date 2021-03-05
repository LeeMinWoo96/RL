import gym
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import sys
sys.path.append('./lib')
import lib

gamma  = 1.0
alpha = 0.5
e = 0.1 # epslion
num_episodes = 300
env = gym.make('WindyGridWorld-v0')
num_actions = env.action_space.n

Q = defaultdict(lambda: np.zeros(num_actions))


episode_length = []

for episode in range(num_episodes):
    s = env.reset()
    len_current_episode = 1
    p = np.random.random()
    if p < e: # 10% 이하 일때 Policy 말고 랜덤 탐색 진행
        a = np.random.choice(num_actions) # 결국 0~3 중 하나 정하기
    else:
        a = np.argmax(Q[s]) # 가장 큰 값의 액션 취한다 
    done = False
    while not done:
        s_ ,r ,done, _ = env.step(a)
        len_current_episode += 1
        p = np.random.random()
        if p < e: # next action 
            a_ = np.random.choice(num_actions) 
        else:
            a_ = np.argmax(Q[s_]) 
        Q[s][a] += alpha * (r + gamma * Q[s_][a_] - Q[s][a])
        s = s_
        a = a_
    episode_length.append(len_current_episode)
fig, ax = plt.subplots(figsize = (8,4))
ax.plot(episode_length)
ax.legend()
plt.show()