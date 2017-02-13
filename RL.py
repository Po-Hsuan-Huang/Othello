#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 22:53:35 2017

@author: pohsuanhuang
"""
import numpy as np

# probability for exploration
EPSILON = 0.1

# step size
ALPHA = 0.5

# gamma for Q-Learning and Expected Sarsa
GAMMA = 1


 # Goal State of the game.
IsEndState  = False

# initial state action pair values
def InitStateActionValues():
      return StateActionValues = np.zeros((states, 64)) # 64 possible place to place the tile.
      
def InitState(state) :
    '''
    parameters
    ----------
    
    state : board
    
    stateActionValues : stateActionValues
            Q(s,a)
    
    '''
    
    # from resetBoard(borad)
     for x in range(8):
         for y in range(8):
             board[x][y] = ' '
     # Starting pieces:
     state[3][3] = 'X'
     state[3][4] = 'O'
     state[4][3] = 'O'
     state[4][4] = 'X'
     
     global moveCounts
     moveCounts = -1
     
     return State 
     

def chooseAction(state, stateActionValues):
    '''
    parameters
    ----------
    
    state : board
    
    stateActionValues : stateActionValues
            Q(s,a)
    
    '''
    rd = np.random.random()
    if nd < EPSILON
        return np.random.choice(Possible_actions)
    else:
        return np.argmax(stateActionValues[state, :])

an episode with Sarsa
# @stateActionValues: values for state action pair, will be updated
# @expected: if True, will use expected Sarsa algorithm
# @stepSize: step size for updating
# @return: total rewards within this episode
def sarsa(stateActionValues, expected=False, stepSize=ALPHA):
    currentState = startState
    currentAction = chooseAction(currentState, stateActionValues)
    rewards = 0.0
    while currentState != goalState:
        newState = actionDestination[currentState][currentAction]
        newAction = chooseAction(newState, stateActionValues)
        reward = actionRewards[currentState, currentAction]
        rewards += reward
        if not expected:
            valueTarget = stateActionValues[newState, newAction]
        else:
            # calculate the expected value of new state
            valueTarget = 0.0
            bestActions = np.argmax(stateActionValues[newState, :], unique=False)
            for action in actions:
                if action in bestActions:
                    valueTarget += ((1.0 - EPSILON) / len(bestActions) + EPSILON / len(actions)) * stateActionValues[newState[0], newState[1], action]
                else:
                    valueTarget += EPSILON / len(actions) * stateActionValues[newState, action]
            valueTarget *= GAMMA
        # Sarsa update
        stateActionValues[currentState, currentAction] += stepSize * (reward +
            valueTarget - stateActionValues[currentState, currentAction])
        currentState = newState
        currentAction = newAction
    return rewards

# an episode with Q-Learning
# @stateActionValues: values for state action pair, will be updated
# @expected: if True, will use expected Sarsa algorithm
# @stepSize: step size for updating
# @return: total rewards within this episode
def qLearning(stateActionValues, stepSize=ALPHA):
    currentState = startState
    rewards = 0.0
    while currentState != goalState:
        currentAction = chooseAction(currentState, stateActionValues)
        reward = actionRewards[currentState, currentAction]
        rewards += reward
        newState = actionDestination[currentState][currentAction]
        # Q-Learning update
        stateActionValues[currentState, currentAction] += stepSize * (
            reward + GAMMA * np.max(stateActionValues[newState, :]) -
            stateActionValues[currentState, currentAction])
        currentState = newState
    return rewards