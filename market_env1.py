from random import random
import numpy as np
import math
import logging # review
import datetime # review

import gym
from gym import spaces

logging.basicConfig(
    level = logging.INFO,
    format = '%(asctime)s - %(filename)s:%(lineno)s - %(name)s - %(message)s',
    datefmt = '%Y-%m-%d %A %H:%M:%S',
) # review
logger = logging.getLogger('market_env') # review

class MarketEnv(gym.Env):
    PENALTY = 1  # 0.999756079
    COST = 0.005

    def __init__(self, dir_path, target_codes, input_codes, start_date, end_date, scope=60, sudden_death=-1.,
                 cumulative_reward=False):
        self.startDate = start_date
        self.endDate = end_date
        self.scope = scope
        self.sudden_death = sudden_death
        self.cumulative_reward = cumulative_reward

        self.inputCodes = []
        self.targetCodes = []
        self.dataMap = {}

        for code in (target_codes + input_codes):
            fn = dir_path + "./" + code + ".csv"

            data = {}
            lastClose = 0
            lastVolume = 0
            lastHigh = 0
            lastLow = 0
            try:
                f = open(fn, "r")
                for line in f:
                    if line.strip() != "":
                        #dt, close, volume, openPrice, high, low = line.strip().split(",")#used for American(except index)
                        dt, openPrice, high, low, close, volume = line.strip().split(",") #used for index
                        dt0 = datetime.datetime.strptime(dt,'%Y/%m/%d') # review
                        dt = datetime.datetime.strftime(dt0,'%Y-%m-%d') # review

                        try:
                            if dt >= start_date:
                                high = float(high) if high != "" else float(close)
                                low = float(low) if low != "" else float(close)
                                close = float(close)
                                volume = float(volume)

                                if lastClose > 0 and close > 0 and lastVolume > 0 and high>0 and lastHigh > 0 and low > 0 and lastLow>0 :
                                    close_ = math.log(close/lastClose)#(close - lastClose) / lastClose
                                    if close_ == 0:
                                        continue
                                    high_ = math.log(high/lastHigh)#(high - close) / close
                                    low_ = math.log(high/lastLow)#(low - close) / close
                                    volume_ = math.log(volume/lastVolume)#(volume - lastVolume) / lastVolume

                                    action_ = 0 if close_ > 0 else 1 # review: right action of tm1 day

                                    data[dt] = (high_, low_, close_, volume_, action_) # review: add action_

                                lastClose = close
                                lastVolume = volume
                                lastHigh = high
                                lastLow = low
                        except Exception, e:
                            print e, line.strip().split(",")
                f.close()
            except Exception, e:
                print e

            if len(data.keys()) > scope:
                self.dataMap[code] = data
                if code in target_codes:
                    self.targetCodes.append(code)
                if code in input_codes:
                    self.inputCodes.append(code)

        self.actions = [
            "LONG",
            "SHORT",
        ]

        self.action_space = spaces.Discrete(len(self.actions))
        self.observation_space = spaces.Box(np.ones(scope * (len(input_codes) + 1)) * -1,
                                            np.ones(scope * (len(input_codes) + 1)))

        self._reset()
        self._seed()

    def _step(self, action):
        if self.done:
            return self.state, self.reward, self.done, self.change, {}, self.trueaction, self.capital_long, self.capital_short # review: add self.trueaction
        #self.lastreward = 0
        #self.newreward = 0

        self.reward = 0
        self.risk = 0 #review
        self.change = False

        vari = self.target[self.targetDates[self.currentTargetIndex]][2]
        #self.cum = self.cum * (1 + vari) # used for discrete interest
        self.cum += vari #used for continuous interest
        self.reward += -MarketEnv.COST

        for i in xrange(len(self.boughts)):
            #self.boughts[i] = self.boughts[i] * MarketEnv.PENALTY * (1 + vari * (-1 if sum(self.boughts) < 0 else 1))#used for discrete interest
            self.Boughts[i] = (self.Boughts[i] + vari * (
                -1 if sum(self.boughts) < 0 else 1)) * MarketEnv.PENALTY  # for continuous interest


        if self.actions[action] == "LONG":
            #for b in self.boughts:
                #self.reward +=(b * (1 if sum(self.boughts) > 0 else -1)) - 1 # used for discrete interest
            for b in self.Boughts:
                self.reward += b #* (1 if sum(self.boughts) > 0 else -1)  # used for continuous interest
            # self.newreward += -MarketEnv.COST
            # self.reward = -self.lastreward + self.newreward

            if self.cumulative_reward:
                self.reward = self.reward / max(1, len(self.boughts))
            #self.lastreward = self.newreward

            if sum(self.boughts) < 0:
                #self.reward += -MarketEnv.COST

                if self.sudden_death * len(self.boughts) > self.reward: #review: if there's too much loss, this path is finished
                    self.done = True

                self.capital_short.append(len(self.boughts))  # capital
                self.boughts = []
                self.Boughts = []
                #self.lastreward = 0
                self.change = True

            self.boughts.append(1.0)

        elif self.actions[action] == "SHORT":
            
            #for b in self.boughts:
                #self.reward += (b * (1 if sum(self.boughts) > 0 else -1)) - 1 #used for discrete interest
            for b in self.Boughts:
                self.reward += b #* (1 if sum(self.boughts) > 0 else -1)  # used for continuous interest
            #self.newreward += - MarketEnv.COST
            #self.reward = -self.lastreward + self.newreward
            if self.cumulative_reward:
                self.reward = self.reward / max(1, len(self.boughts))
            #self.lastreward = self.newreward


            if sum(self.boughts) > 0:
                #self.reward += -MarketEnv.COST
                if self.sudden_death * len(self.boughts) > self.reward: #review: if there's too much loss, this path is finished
                    self.done = True

                self.capital_long.append(len(self.boughts))
                self.boughts = []
                self.Boughts = []
                #self.lastreward = 0
                self.change = True

            self.boughts.append(-1.0)
        else:
            pass

        self.Boughts.append(0.0)
       
        self.defineState()
        self.trueaction = self.target[self.targetDates[self.currentTargetIndex]][
            4]  # review: right action of tm1 day

        self.currentTargetIndex += 1
        if self.currentTargetIndex >= len(self.targetDates) or self.endDate <= self.targetDates[

            self.currentTargetIndex]:
            self.done = True

        if self.done:
            #for b in self.boughts:
                #self.reward += (b * (1 if sum(self.boughts) > 0 else -1)) - 1 - MarketEnv.COST #review:transaction cost added
            #if self.cumulative_reward:
                #self.reward = self.reward / max(1, len(self.boughts))

            if sum(self.boughts) > 0:
                self.capital_long.append(len(self.boughts))
            elif sum(self.boughts) < 0:
                self.capital_short.append(len(self.boughts))
            self.boughts = []
            self.Boughts = []

        #print self.targetDates[self.currentTargetIndex-1], "date" # review: !!!!test
        #print self.trueaction, "rightaction", action, "action", vari, "vari" # review: !!!test
        #self.reward += -self.lastreward
        #self.lastreward = self.reward


        return self.state, self.reward, self.done, self.change, {"dt": self.targetDates[self.currentTargetIndex-1], "cum": self.cum,
                                                    "code": self.targetCode}, self.trueaction, self.capital_long, self.capital_short # review: add right action of t(current) day



    def _steptest(self, action):
        if self.done:
            return self.state, self.reward, self.done, self.change, {}, self.trueaction, self.capital_long, self.capital_short # review: add self.trueaction
        #self.lastreward = 0
        #self.newreward = 0

        self.reward = 0
        self.risk = 0 #review
        self.change = False
        self.reward += -MarketEnv.COST

        vari = self.target[self.targetDates[self.currentTargetIndex]][2]
        #self.cum = self.cum * (1 + vari)#used for discrete interest
        self.cum += vari #used for continuous interest

        for i in xrange(len(self.boughts)):
            #self.boughts[i] = self.boughts[i] * MarketEnv.PENALTY * (1 + vari * (-1 if sum(self.boughts) < 0 else 1))# used for discrete interest
            self.Boughts[i] = (self.Boughts[i] + vari * (
                -1 if sum(self.boughts) < 0 else 1)) * MarketEnv.PENALTY  # for continuous interest

        if self.actions[action] == "LONG":
            #for b in self.boughts:
                #self.reward +=(b * (1 if sum(self.boughts) > 0 else -1)) - 1#used for discrete interest
            for b in self.Boughts:
                self.reward += b #* (1 if sum(self.boughts) > 0 else -1)  # used for continuous interest
            #self.newreward += -MarketEnv.COST
            #self.reward = -self.lastreward + self.newreward
	    
            if self.cumulative_reward:
                self.reward = self.reward / max(1, len(self.boughts))
            #self.lastreward = self.newreward

            if sum(self.boughts) < 0:
                #self.reward += -MarketEnv.COST

                if self.sudden_death * len(self.boughts) > self.reward: #review: if there's too much loss, this path is finished
                    self.done = True

                self.capital_short.append(len(self.boughts))
                self.boughts = []
                self.Boughts = []
                #self.lastreward = 0
                self.change = True

            self.boughts.append(1.0)

        elif self.actions[action] == "SHORT":
            
            #for b in self.boughts:
                #self.reward += (b * (1 if sum(self.boughts) > 0 else -1)) - 1 #used for discrete interest
            for b in self.Boughts:
                self.reward += b #* (1 if sum(self.boughts) > 0 else -1)  # used for continuous interest
            #self.newreward += - MarketEnv.COST
            #self.reward = -self.lastreward + self.newreward
            if self.cumulative_reward:
                self.reward = self.reward / max(1, len(self.boughts))
            #self.lastreward = self.newreward

	    
            if sum(self.boughts) > 0:
                #self.reward += -MarketEnv.COST
                if self.sudden_death * len(self.boughts) > self.reward: #review: if there's too much loss, this path is finished
                    self.done = True

                self.capital_long.append(len(self.boughts))
                self.boughts = []
                self.Boughts = []
                #self.lastreward = 0
                self.change = True

            self.boughts.append(-1.0)
        else:
            pass

        self.Boughts.append(0.0)
        self.defineState()
        self.trueaction = self.target[self.targetDates[self.currentTargetIndex]][
            4]  # review: right action of tm1 day

        self.currentTargetIndex += 1
        if self.currentTargetIndex >= len(self.targetDates) or self.endDate <= self.targetDates[
            self.currentTargetIndex]:
            self.done = True

        if self.done:
            #for b in self.boughts:
                #self.reward += (b * (1 if sum(self.boughts) > 0 else -1)) - 1 - MarketEnv.COST #review:transaction cost added
            #if self.cumulative_reward:
                #self.reward = self.reward / max(1, len(self.boughts))
            if sum(self.boughts) > 0:
                self.capital_long.append(len(self.boughts))
            elif sum(self.boughts) < 0:
                self.capital_short.append(len(self.boughts))

            self.boughts = []
            self.Boughts = []

        #print self.targetDates[self.currentTargetIndex-1], "date" # review: !!!!test
        #print self.trueaction, "rightaction", action, "action", vari, "vari" # review: !!!test
        #self.reward += -self.lastreward
        #self.lastreward = self.reward


        return self.state, self.reward, self.done, self.change, {"dt": self.targetDates[self.currentTargetIndex-1], "cum": self.cum,
                                                    "code": self.targetCode}, self.trueaction, self.capital_long, self.capital_short # review: add right action of t(current) day

    def _reset(self):
        self.targetCode = self.targetCodes[int(random() * len(self.targetCodes))]
        logger.info("select stock %s " % (self.targetCode))
        self.target = self.dataMap[self.targetCode]
        self.targetDates = sorted(self.target.keys())
        self.currentTargetIndex = self.scope
        self.boughts = []
        self.Boughts = []
        self.capital_long = []
        self.capital_short  =[]
        self.cum = 1.

        self.done = False
        self.change = False
        self.reward = 0
        self.trueaction = self.target[self.targetDates[self.currentTargetIndex]][4] # review: right action of tm1 day

        self.defineState()

        return self.state, self.trueaction # review: right action of tm1 day

    def _resettest(self,i):
        self.targetCode = self.targetCodes[i]
        logger.info("select stock %s " % (self.targetCode))
        self.target = self.dataMap[self.targetCode]
        self.targetDates = sorted(self.target.keys())
        self.currentTargetIndex = self.scope
        self.boughts = []
        self.Boughts = []
        self.capital_long = []
        self.capital_short = []
        self.cum = 1.

        self.done = False
        self.change = False
        self.reward = 0
        self.trueaction = self.target[self.targetDates[self.currentTargetIndex]][4] # review: right action of tm1 day

        self.defineState()

        return self.state, self.trueaction # review: right action of tm1 day

    def _render(self, mode='human', close=False):
        if close:
            return
        return self.state

    '''
	def _close(self):
		pass

	def _configure(self):
		pass
	'''

    def _seed(self):
        return int(random() * 100)

    def defineState(self):
        tmpState = []

        budget = (sum(self.boughts) / len(self.boughts)) if len(self.boughts) > 0 else 1.
        size = math.log(max(1., len(self.boughts)), 100)
        position = 1. if sum(self.boughts) > 0 else 0.
        tmpState.append([[budget, size, position]])

        subject = []
        subjectVolume = []
        subjecthigh = [] # review: high_
        subjectlow = [] # review: low_
        for i in xrange(self.scope): #review: each state has self.scope days' transaction information
            try:
                subjecthigh.append([self.target[self.targetDates[self.currentTargetIndex - 1 - i]][0]]) #review: high_
                subjectlow.append([self.target[self.targetDates[self.currentTargetIndex - 1 - i]][1]]) #review: low_
                subject.append([self.target[self.targetDates[self.currentTargetIndex - 1 - i]][2]]) #review: close_
                subjectVolume.append([self.target[self.targetDates[self.currentTargetIndex - 1 - i]][3]]) #review: volume_
            except Exception, e:
                print self.targetCode, self.currentTargetIndex, i, len(self.targetDates)
                self.done = True
        tmpState.append([[subjecthigh, subjectlow, subject, subjectVolume]])

        tmpState = [np.array(i) for i in tmpState]
        self.state = tmpState #review: self.state = array([[budget,size,position,],[subject, subjectVolume]])
