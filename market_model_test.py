from market_env1 import MarketEnv
from market_model_builder import MarketModelBuilder

import logging # review

import sys
import codecs
import numpy as np

#logger1 = logging.getLogger('market_dqn')  # review
#log_file1 = logging.FileHandler('test_result_600060.log')  # review
#logger.addHandler(self.log_file1)  # review
#logging.Logger.propagate = False

def Lgr(message):
    logging.basicConfig(
        level = logging.INFO,
        format = '%(asctime)s - %(filename)s:%(lineno)s - %(name)s - %(message)s',
        datefmt = '%Y-%m-%d %A %H:%M:%S',
    ) # review
    logger = logging.getLogger('market_model_test') # review
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

class ModelEvaluate(object):
    def __init__(self,Numtest = 12, codeListFilename = None, dir_path = "./testdata/", start_date = "2018-03-07",
                 end_date = "2018-09-24", max_memory = 18, batch_size = 18, discount = 0.8, n_steps = 5,
                 backpropagate = False, modelname = "GSPC.h5"):
        self.Numtest = Numtest
        self.codeListFilename = codeListFilename
        self.dir_path = dir_path
        self.start_date = start_date
        self.end_date = end_date
        self.max_memory = max_memory  # review: original 5000
        self.batch_size = batch_size
        self.discount = discount
        self.n_steps = n_steps
        self.backpropagate = backpropagate
        self.modelname = modelname
        #logging.basicConfig(
            #level=logging.INFO,
            #format='%(asctime)s - %(filename)s:%(lineno)s - %(name)s - %(message)s',
            #datefmt='%Y-%m-%d %A %H:%M:%S',
        #)  # review
        #logging.Logger.propagate = False # review
        #self.logger1 = logging.getLogger('market_dqn')  # review
        #self.log_file1 = logging.FileHandler('test_result_600060.log')  # review
        #self.logger1.addHandler(self.log_file1)  # review

    def _readfile(self):

        codeMap = {}
        f = codecs.open(self.codeListFilename, "r", "utf-8")

        for line in f:

            if line.strip() != "":
                tokens = line.strip().split(",") if not "\t" in line else line.strip().split("\t")
                codeMap[tokens[0]] = tokens[1]

        f.close()  # read data

        self.env = MarketEnv(dir_path=self.dir_path, target_codes=codeMap.keys(), input_codes=[], start_date=self.start_date,
                        end_date=self.end_date, sudden_death=-1.0)  # set environment

    def _loadmodel(self,model = None, model1 = None,load = False): # review: !!!!!!!!!!!!!*************

        from market_dqn import ExperienceReplay

        if load == True:

            self.model = MarketModelBuilder().getModel
            # target action-value function Q~********************
            self.model1 = MarketModelBuilder().getModel
            #sgd = SGD(lr=0.0009, decay=1e-6, momentum=0.9, nesterov=True)  # review: !!must exist evern when loading weights aiming to update, else not.
            self.model.compile(loss='mse',
                          optimizer='rmsprop')  # review: !!must exist even when loading weights aiming to update, else not.

            self.model.load_weights(self.modelname)  # load saved model

            #sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)  # review: !!must exist evern when loading weights
            self.model1.compile(loss='mse', optimizer='rmsprop')  # review: !!must exist even when loading weights
            self.model1.set_weights(self.model.get_weights())

        else:
            self.model = model
            self.model1 = model1

        # Initialize experience replay object
        self.exp_replay = ExperienceReplay(max_memory=self.max_memory, discount=self.discount)

    def _test(self):

        for i in xrange(self.Numtest):
            epsilon = 0.0
            loss = 0.0

            cumReward = 0
            lastreward = 0
            game_over = False
            change = False
            change_tm1 = False
            input_t, rightaction = self.env._resettest(i)  # review: add rightaction
            number_total = 0  # review:
            number_dl = 0  # review:
            Rightnumber_total = 0  # review:
            Rightnumber_dl = 0  # review:
            accuracy_total = 0.0  # review:
            accuracy_dl = 0.0  # review:
            capital_long = 1
            capital_short = 1
            capitallist_long = []
            capitallist_short = []
            TP = 0.0  # review: true long
            FP = 0.0  # review: false short
            TN = 0.0  # review: true short
            FN = 0.0  # review: false long
            ori_matrix = [[TP, TN], [FP, FN]]  # review: original statistics matrix for F1_score

            Lgr("*********")

            while not game_over:
		
                input_tm1 = input_t
                isRandom = False
                number_total += 1  # review:

                if np.random.rand() <= epsilon:
                    action = np.random.randint(0, self.env.action_space.n, size=1)[0]
                    input_t, cumpolicyreward, game_over, change, info, rightaction, capitallist_long, capitallist_short = self.env._steptest(
                        action)  # review: input_t = self.state in market_env

                    isRandom = True
                else:
                    q = self.model.predict(input_tm1)
                    action = np.argmax(q[0])

                    # print "  ".join(["%s:%.2f" % (l, i) for l, i in zip(env.actions, q[0].tolist())])
                    if np.nan in q:
                        print "OCCUR NaN!!!"
                        exit()

                    input_t, cumpolicyreward, game_over, change, info, rightaction, capitallist_long, capitallist_short = self.env._steptest(
                        action)  # review: input_t = self.state in market_env
                    number_dl += 1  # review:
                    if action == rightaction:
                        Rightnumber_dl += 1  # review:
                    accuracy_dl = float(Rightnumber_dl) / float(number_dl)  # review:

                if change_tm1:#change:
                    reward = cumpolicyreward
                else:
                    reward = cumpolicyreward - lastreward

                cumReward += reward

                lastreward = cumpolicyreward
                change_tm1 = change
                if action == rightaction:
                    Rightnumber_total += 1  # review:
                    ori_matrix[0][action] += 1  # review: compute the trueactions (right long and right short)
                else:
                    ori_matrix[1][action] += 1  # review: compute the actions (long and short)

                accuracy_total = float(Rightnumber_total) / float(number_total)  # review:

                print action, rightaction  # review: !!!!test
                print number_dl, Rightnumber_dl, number_total, Rightnumber_total  # review: !!!! test


                if self.env.actions[action] == "LONG" or self.env.actions[action] == "SHORT":
                    color = bcolors.FAIL if self.env.actions[action] == "LONG" else bcolors.OKBLUE
                    if isRandom:
                        color = bcolors.WARNING if self.env.actions[action] == "LONG" else bcolors.OKGREEN
                        # print "%s:\t%s\t%.2f\t%.2f\t" % (
                        # info["dt"], color + env.actions[action] + bcolors.ENDC, cumReward, info["cum"]) + ("\t".join(
                        # ["%s:%.2f" % (l, i) for l, i in zip(env.actions, q[0].tolist())]) if isRandom == False else "")
                    
                    Lgr("%s:\t%s\t%.4f\t%.4f\t" % (
                        info["dt"], color + self.env.actions[action] + bcolors.ENDC, cumReward, info["cum"]) + ("\t".join(
                        ["%s:%.2f" % (l, i) for l, i in
                         zip(self.env.actions, q[0].tolist())]) if isRandom == False else ""))  # review
                    Lgr(
                        "accuracy_dl: %.4f, accuracy_total: %.4f" % (
                        accuracy_dl, accuracy_total))  # review: logger accuracy

                if self.backpropagate == True:
                    # store experience
                    self.exp_replay.remember([input_tm1, action, reward, input_t], game_over)

                    # adapt model
                    inputs, targets = self.exp_replay.get_batch(self.model, self.model1, batch_size=self.batch_size)

                    loss += self.model.train_on_batch(inputs, targets)

                    if number_total % self.n_steps == 0:
                        self.model1.set_weights(self.model.get_weights())

            capital_long = max(capitallist_long) if len(capitallist_long) != 0 else 0
            capital_short = max(capitallist_short) if len(capitallist_short) != 0 else 0

            Lgr("TP: % .2f, TN: % .2f, FP: % .2f, FN: % .2f" % (
            ori_matrix[0][0], ori_matrix[0][1], ori_matrix[1][0], ori_matrix[1][1]))  # review
            Lgr("capital_long: %s, capital_short: %s" % (capital_long, capital_short))
            Lgr("*********")
            
            print cumReward, " cumReward"
        #self.logger1.removeHandler(self.log_file1) # review: used for 2 .log files




#if __name__ == "__main__": # review! for test only


    #codeListFilename = sys.argv[1] # review! for test only

    '''codeMap = {}
    f = codecs.open(codeListFilename, "r", "utf-8")


    for line in f:

        if line.strip() != "":
            tokens = line.strip().split(",") if not "\t" in line else line.strip().split("\t")
            codeMap[tokens[0]] = tokens[1]

    f.close() # read data

    env = MarketEnv(dir_path="./testdata/", target_codes=codeMap.keys(), input_codes=[], start_date="2015-08-26",
                    end_date="2016-03-31", sudden_death=-1.0) # set environment

    max_memory = 18  # review: original 5000
    batch_size = 18
    discount = 0.8
    n_steps = 5
    backpropagate = True'''


    #from keras.optimizers import SGD
    '''model = MarketModelBuilder().getModel
    sgd = SGD(lr=0.0009, decay=1e-6, momentum=0.9, nesterov=True)  # review: !!must exist evern when loading weights aiming to update, else not.
    model.compile(loss='mse', optimizer='rmsprop')  # review: !!must exist even when loading weights aiming to update, else not.

    model.load_weights("dqntest_600060.h5") # load saved model

    # target action-value function Q~********************
    model1 = MarketModelBuilder().getModel
    sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)  # review: !!must exist evern when loading weights
    model1.compile(loss='mse', optimizer='rmsprop')  # review: !!must exist even when loading weights
    model1.set_weights(model.get_weights())

    # Initialize experience replay object
    exp_replay = ExperienceReplay(max_memory=max_memory, discount=discount)'''


    #*********************evaluate************************* #
    #modeltest = ModelEvaluate(codeListFilename=codeListFilename, Numtest=1) # review! for test only
    #modeltest._readfile() # review! for test only
    #modeltest._loadmodel() # review! for test only
    #modeltest._test() # review! for test only

    '''for i in xrange(10):
        epsilon = 0.0
        loss = 0.0

        cumReward = 0
        game_over = False
        input_t, rightaction = env._resettest(i) # review: add rightaction
        number_total = 0  # review:
        number_dl = 0  # review:
        Rightnumber_total = 0  # review:
        Rightnumber_dl = 0  # review:
        accuracy_total = 0.0  # review:
        accuracy_dl = 0.0  # review:
        TP = 0.0  # review: true long
        FP = 0.0  # review: false short
        TN = 0.0  # review: true short
        FN = 0.0  # review: false long
        ori_matrix = [[TP, TN],[FP, FN]] # review: original statistics matrix for F1_score

        while not game_over:

            input_tm1 = input_t
            isRandom = False
            number_total += 1  # review:

            if np.random.rand() <= epsilon:
                action = np.random.randint(0, env.action_space.n, size= 1)[0]
                input_t, reward, game_over, info, rightaction = env._step(action)  # review: input_t = self.state in market_env

                isRandom = True
            else:
                q = model.predict(input_tm1)
                action = np.argmax(q[0])

                # print "  ".join(["%s:%.2f" % (l, i) for l, i in zip(env.actions, q[0].tolist())])
                if np.nan in q:
                    print "OCCUR NaN!!!"
                    exit()

                input_t, reward, game_over, info, rightaction = env._step(
                    action)  # review: input_t = self.state in market_env
                number_dl += 1  # review:
                if action == rightaction:
                    Rightnumber_dl += 1  # review:
                accuracy_dl = float(Rightnumber_dl) / float(number_dl)  # review:

            cumReward += reward
            if action == rightaction:
                Rightnumber_total += 1  # review:
                ori_matrix[0][action] += 1 # review: compute the trueactions (right long and right short)
            else:
                ori_matrix[1][action] += 1 # review: compute the actions (long and short)

            accuracy_total = float(Rightnumber_total) / float(number_total)  # review:

            print action, rightaction  # review: !!!!test
            print number_dl, Rightnumber_dl, number_total, Rightnumber_total  # review: !!!! test

            if env.actions[action] == "LONG" or env.actions[action] == "SHORT":
                color = bcolors.FAIL if env.actions[action] == "LONG" else bcolors.OKBLUE
                if isRandom:
                    color = bcolors.WARNING if env.actions[action] == "LONG" else bcolors.OKGREEN
                #print "%s:\t%s\t%.2f\t%.2f\t" % (
                    #info["dt"], color + env.actions[action] + bcolors.ENDC, cumReward, info["cum"]) + ("\t".join(
                    #["%s:%.2f" % (l, i) for l, i in zip(env.actions, q[0].tolist())]) if isRandom == False else "")
                logger.info("%s:\t%s\t%.4f\t%.4f\t" % (
                    info["dt"], color + env.actions[action] + bcolors.ENDC, cumReward, info["cum"]) + ("\t".join(
                    ["%s:%.2f" % (l, i) for l, i in zip(env.actions, q[0].tolist())]) if isRandom == False else ""))  # review
                logger.info(
                    "accuracy_dl: %.4f, accuracy_total: %.4f" % (accuracy_dl, accuracy_total))  # review: logger accuracy

            if backpropagate == True:
                # store experience
                exp_replay.remember([input_tm1, action, reward, input_t], game_over)

                # adapt model
                inputs, targets = exp_replay.get_batch(model, model1, batch_size=batch_size)

                loss += model.train_on_batch(inputs, targets)

                if number_total % n_steps == 0:
                    model1.set_weights(model.get_weights())

        logger.info("TP: % .2f, TN: % .2f, FP: % .2f, FN: % .2f" % (ori_matrix[0][0], ori_matrix[0][1], ori_matrix[1][0], ori_matrix[1][1])) # review

        print cumReward ," cumReward"'''
