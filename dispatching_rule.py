from Environment_Multi.FJSP import RL_ENV

env = RL_ENV(mode = 'ssu')
env.render()

while True:
    action = [1]
    reward, done, actions = env.step(action)
    if done==True:
        print(env.env.now)
        break
