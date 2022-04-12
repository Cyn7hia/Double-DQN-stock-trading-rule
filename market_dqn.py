import numpy as np
import logging # review

from market_env1 import MarketEnv
from market_model_builder import MarketModelBuilder
import os
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"]="0"
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9
session = tf.Session(config=config)

def Lgr(message):
    logging.basicConfig(
        level = logging.INFO,
        format = '%(asctime)s - %(filename)s:%(lineno)s - %(name)s - %(message)s',
        datefmt = '%Y-%m-%d %A %H:%M:%S',
    ) # review
    logger = logging.getLogger('market_dqn') # review
    log_file = logging.FileHandler('GSPC.log') # review
    logger.addHandler(log_file) # review
    logger.info(message)
    logger.removeHandler(log_file)

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class ExperienceReplay(object):
    def __init__(self, max_memory=100, discount=.9):
        self.max_memory = max_memory
        self.memory = list()
        self.discount = discount

    def remember(self, states, game_over):
        # memory[i] = [[state_t, action_t, reward_t, state_t+1], game_over?]
        self.memory.append([states, game_over])
        if len(self.memory) > self.max_memory:
            del self.memory[0]

    def get_batch(self, model, model1, batch_size=10):  #review: gradient of Q-learning
        len_memory = len(self.memory)
        num_actions = model.output_shape[-1]
        inputs = []

        dim = len(self.memory[0][0][0])
        for i in xrange(dim):
            inputs.append([])

        targets = np.zeros((min(len_memory, batch_size), num_actions))
        for i, idx in enumerate(np.random.randint(0, len_memory, size=min(len_memory, batch_size))):
            state_t, action_t, reward_t, state_tp1 = self.memory[idx][0]
            game_over = self.memory[idx][1]

            for j in xrange(dim):
                inputs[j].append(state_t[j][0])

            # inputs.append(state_t)
            # There should be no target values for actions not taken.
            # Thou shalt not correct actions not taken #deep
            targets[i] = model1.predict(state_tp1)[0]  #review: Q_sa
            q = model.predict(state_tp1)
            action = np.argmax(q[0])
            Q_sa = targets[i][action] # review: targets_DoubleDQN_version
            #Q_sa = np.max(model1.predict(state_tp1)[0])  #review: targets_version1
            if game_over:  # if game_over is True
                targets[i, action_t] = reward_t
            else:
                # reward_t + gamma * max_a' Q(s', a')
                targets[i, action_t] = reward_t + self.discount * Q_sa

        # inputs = np.array(inputs)
        inputs = [np.array(inputs[i]) for i in xrange(dim)]

        return inputs, targets  # inputs, Q_sa


if __name__ == "__main__":
    import sys
    import codecs
    from market_model_test import ModelEvaluate

    codeListFilename = sys.argv[1]
    modelFilename = sys.argv[2] if len(sys.argv) > 2 else None

    codeMap = {}
    f = codecs.open(codeListFilename, "r", "utf-8")

    for line in f:
        if line.strip() != "":
            tokens = line.strip().split(",") if not "\t" in line else line.strip().split("\t")
            codeMap[tokens[0]] = tokens[1]

    f.close()

    env = MarketEnv(dir_path="./data/", target_codes=codeMap.keys(), input_codes=[], start_date="2014-10-06",
                    end_date="2018-05-31", sudden_death=-1.0)

    # parameters
    epsilon = .5  # exploration
    min_epsilon = 0.1
    epoch = 100 #review: original 100000
    max_memory = 700 #review: original 5000
    batch_size = 50
    discount = 0.9
    n_steps = 10

    from keras.optimizers import SGD
    # action-value function Q
    model = MarketModelBuilder(modelFilename).getModel
    sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True) # review: !!must exist evern when loading weights
    model.compile(loss='mse', optimizer='rmsprop') # review: !!must exist even when loading weights

    #model.load_weights("dqntest.h5")  # load saved model # review: !!load unfinished model

    # target action-value function Q~********************
    model1 = MarketModelBuilder(modelFilename).getModel
    sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)  # review: !!must exist evern when loading weights
    model1.compile(loss='mse', optimizer='rmsprop')  # review: !!must exist even when loading weights
    model1.set_weights(model.get_weights())

    # Initialize experience replay object
    exp_replay = ExperienceReplay(max_memory=max_memory, discount=discount)

    # Train
    win_cnt = 0
    for e in range(epoch):
        loss = 0.
        env._reset()
        game_over = False
        change = False #change_tm1
        change_tm1 = False #change_tm2
        # get initial input
        input_t, rightaction = env._reset() # review: add rightaction
        cumReward = 0
        lastreward = 0
        number_total = 0 # review:
        number_dl = 0 # review:
        Rightnumber_total = 0 # review:
        Rightnumber_dl = 0 # review:
        accuracy_total = 0.0 # review:
        accuracy_dl = 0.0 # review:
        capital_long = 1
        capital_short = 1
        capitallist_long = []
        capitallist_short = []
        #TP = 0.0  # review: true long
        #FP = 0.0  # review: false short
        #TN = 0.0  # review: true short
        #FN = 0.0  # review: false long
        #ori_matrix = [[TP, TN],[FP, FN]] # review: original statistics matrix for F1_score

        while not game_over: #review: game_over = self.done in market_env
            input_tm1 = input_t
            isRandom = False
            number_total += 1 # review:

            # get next action
            if np.random.rand() <= epsilon:
                action = np.random.randint(0, env.action_space.n, size=1)[0]
                input_t, cumpolicyreward, game_over, change, info, rightaction, capitallist_long, capitallist_short = env._step(
                    action)  # review: input_t = self.state in market_env

                isRandom = True
            else:
                q = model.predict(input_tm1)
                action = np.argmax(q[0])

                # print "  ".join(["%s:%.2f" % (l, i) for l, i in zip(env.actions, q[0].tolist())])
                if np.nan in q:
                    print "OCCUR NaN!!!"
                    exit()

                input_t, cumpolicyreward, game_over, change, info, rightaction, capitallist_long, capitallist_short = env._step(
                    action)  # review: input_t = self.state in market_env
                number_dl += 1  # review:
                if action == rightaction:
                    Rightnumber_dl += 1 # review:
                accuracy_dl = float(Rightnumber_dl) / float(number_dl)  # review:

            # apply action, get rewards and new state
            #input_t, reward, game_over, info, rightaction = env._step(action) #review: original: input_t = self.state in market_env
            if change_tm1:#change:#change_tm1:
                reward = cumpolicyreward #cumpolicyreward_tm1
            else:
                reward = cumpolicyreward- lastreward
            cumReward += reward

            lastreward = cumpolicyreward
            change_tm1 = change
            if action == rightaction:
                Rightnumber_total += 1 # review:
                #ori_matrix[0][action] += 1 # review: compute the trueactions (right long and right short)
            #else:
                #ori_matrix[1][action] += 1 # review: compute the actions (long and short)

            accuracy_total = float(Rightnumber_total)/float(number_total) # review:

            print action, rightaction # review: !!!!test
            print number_dl, Rightnumber_dl, number_total, Rightnumber_total # review: !!!! test

            if env.actions[action] == "LONG" or env.actions[action] == "SHORT":
                color = bcolors.FAIL if env.actions[action] == "LONG" else bcolors.OKBLUE
                if isRandom:
                    color = bcolors.WARNING if env.actions[action] == "LONG" else bcolors.OKGREEN
                print "%s:\t%s\t%.2f\t%.2f\t" % (
                info["dt"], color + env.actions[action] + bcolors.ENDC, cumReward, info["cum"]) + ("\t".join(
                    ["%s:%.2f" % (l, i) for l, i in zip(env.actions, q[0].tolist())]) if isRandom == False else "") # review: original
                #logger2.info("%s:\t%s\t%.4f\t%.4f\t" % (
                #info["dt"], color + env.actions[action] + bcolors.ENDC, cumReward, info["cum"]) + ("\t".join(
                    #["%s:%.2f" % (l, i) for l, i in zip(env.actions, q[0].tolist())]) if isRandom == False else "")) # review
                #logger2.info("accuracy_dl: %.4f, accuracy_total: %.4f" % (accuracy_dl, accuracy_total )) # review: logger accuracy
                print "accuracy_dl: %.4f, accuracy_total: %.4f" % (
                accuracy_dl, accuracy_total)  # review: logger accuracy

            # store experience
            exp_replay.remember([input_tm1, action, reward, input_t], game_over)

            # adapt model
            inputs, targets = exp_replay.get_batch(model, model1, batch_size=batch_size)

            loss += model.train_on_batch(inputs, targets)

            if number_total % n_steps == 0:
                model1.set_weights(model.get_weights())

        capital_long = max(capitallist_long) if len(capitallist_long) != 0 else 0
        capital_short = max(capitallist_short) if len(capitallist_short) != 0 else 0

        modelevalue = ModelEvaluate(codeListFilename=codeListFilename, Numtest=1)
        modelevalue._readfile()
        modelevalue._loadmodel(model=model, model1=model1, load=False)
        modelevalue._test()

        if cumReward > 0 and game_over:
            win_cnt += 1


        #print("Epoch {:03d}/{} | Loss {:.4f} | Win count {} | Epsilon {:.4f}".format(e, epoch, loss, win_cnt, epsilon)) # review: orginal
        Lgr("Epoch {:03d}/{} | Loss {:.4f} | Win count {} | Epsilon {:.4f}".format(e, epoch, loss, win_cnt, epsilon)) # review
        Lgr(
            "accuracy_dl: %.4f, accuracy_total: %.4f" % (accuracy_dl, accuracy_total))  # review: logger accuracy
        Lgr("capital_long: %s, capital_short: %s" % (capital_long, capital_short))
        # Save trained model weights and architecture, this will be used by the visualization code
        model.save_weights("model.h5" if modelFilename == None else modelFilename, overwrite=True)
        epsilon = max(min_epsilon, epsilon * 0.99)

    Lgr("epsilon: %ep" %epsilon)
