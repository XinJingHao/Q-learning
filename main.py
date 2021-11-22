import gym
import time
import numpy as np
from Q_learning import QLearningAgent
from datetime import datetime
import os, shutil
from torch.utils.tensorboard import SummaryWriter


def evaluate_policy(env, model, render, steps_per_epoch=100):
    s, done, ep_r, steps = env.reset(), False, 0, 0
    while not (done or (steps >= steps_per_epoch)):
        # Take deterministic actions at test time
        a = model.predict(s)
        s_prime, r, done, info = env.step(a)

        ep_r += r
        steps += 1
        s = s_prime
        if render:
            env.render()
    return ep_r

def main():

    write = True #Use SummaryWriter or not
    Loadmodel = False #Load model or not

    EnvName = "CliffWalking-v0"
    env = gym.make(EnvName)
    Env_With_dw = True #Env Like CliffWalking has dw(die&win) signal
    eval_env = gym.make(EnvName)
    max_e_steps = 500 #max episode steps

    Max_train_steps = 20000
    save_interval = 1e10 #in steps
    eval_interval = 100 #in steps

    random_seed = 0
    print("Random Seed: {}".format(random_seed))
    env.seed(random_seed)
    eval_env.seed(random_seed)
    np.random.seed(random_seed)

    if write:
        #Use SummaryWriter to record the trainig
        timenow = str(datetime.now())[0:-7]
        timenow = ' ' + timenow[0:13] + '_' + timenow[14:16]+ '_' + timenow[-2::]
        writepath = 'runs/{}'.format(EnvName) + timenow
        if os.path.exists(writepath): shutil.rmtree(writepath)
        writer = SummaryWriter(log_dir=writepath)

    ''' ↓↓↓ Build Q-learning Agent ↓↓↓ '''
    if not os.path.exists('model'): os.mkdir('model')
    model = QLearningAgent(
        env_with_dw=Env_With_dw,
        s_dim=env.observation_space.n,
        a_dim=env.action_space.n,
        lr=0.2,
        gamma=0.9,
        exp_noise=0.1)
    if Loadmodel: model.restore()

    ''' ↓↓↓ Iterate and Train ↓↓↓ '''
    total_steps = 0
    while total_steps < Max_train_steps:
        s, done, steps = env.reset(), False, 0

        while not (done or steps>max_e_steps):
            steps += 1
            a = model.select_action(s)
            s_, r, done, _ = env.step(a)

            model.train(s, a, r, s_, done)
            s = s_

            '''record & log'''
            if total_steps % eval_interval == 0:
                score = evaluate_policy(eval_env, model, False)
                if write:
                    writer.add_scalar('ep_r', score, global_step=total_steps)
                print('EnvName:',EnvName,'seed:',random_seed,'steps: {}'.format(total_steps),'score:', score)
            total_steps += 1

            '''save model'''
            if total_steps % save_interval==0:
                model.save()
    env.close()

if __name__ == '__main__':
    main()
