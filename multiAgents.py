# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        newGhostPos = successorGameState.getGhostPositions()
        newScore = successorGameState.getScore()
        newNumFood = successorGameState.getNumFood()
        newWin = successorGameState.isWin()

        disghostlist = []
        for ghost in newGhostPos:
            disghostlist += [manhattanDistance(newPos, ghost)]
        disghost = min(disghostlist)

        foodlist = [] 
        for food in newFood.asList():
            foodlist += [manhattanDistance(newPos, food)]
        disfood = 0
        if len(foodlist) != 0:  
            disfood = min(foodlist)

        if sum(newScaredTimes) > 0: 
            disghost = float('inf')
        return 1/(disghost+1.5)*disghost - newNumFood - 0.5/(disfood + 1) * disfood + newScore




def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def value(self, gameState, agentIndex, depth):
        if agentIndex == (gameState.getNumAgents()): 
            agentIndex = 0
        if gameState.isLose():
            return ((self.evaluationFunction(gameState), None))
        if gameState.isWin():
            return ((self.evaluationFunction(gameState), None))
        if agentIndex == 0:
            depth += 1 
        if depth > self.depth: 
            return ((self.evaluationFunction(gameState), None))
        if agentIndex == 0:
            return self.maxvalue(gameState, agentIndex, depth)
                
        else:
            
            return self.minvalue(gameState, agentIndex, depth)

    def minvalue(self, gameState, agentIndex, depth):
        v = (float('inf'), None)
        successorindex = agentIndex + 1
        for action in gameState.getLegalActions(agentIndex):
            successor = gameState.generateSuccessor(agentIndex, action)
            v = min([v, (self.value(successor, successorindex, depth)[0], action)], key = lambda x: x[0])
        agentIndex += 1
        return v 

    def maxvalue(self, gameState, agentIndex, depth):
        v = (-float('inf'), None)
        successorindex = 1
        for action in gameState.getLegalActions(agentIndex):
            successor = gameState.generateSuccessor(agentIndex, action) 
            v = max([v, (self.value(successor, successorindex, depth)[0], action)], key = lambda x: x[0])
        return v 


    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        agentIndex = self.index
        depth = 0
        return self.value(gameState, agentIndex, depth)[1]

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """
    def value(self, gameState, agentIndex, depth, alpha, beta):
        if agentIndex == (gameState.getNumAgents()): 
            agentIndex = 0
        if gameState.isLose():
            return ((self.evaluationFunction(gameState), None))
        if gameState.isWin():
            return ((self.evaluationFunction(gameState), None))
        if agentIndex == 0:
            depth += 1 
        if depth > self.depth: 
            return ((self.evaluationFunction(gameState), None))
        if agentIndex == 0:
            return self.maxvalue(gameState, agentIndex, depth, alpha, beta)
                
        else:
            
            return self.minvalue(gameState, agentIndex, depth, alpha, beta)

    def minvalue(self, gameState, agentIndex, depth, alpha, beta):
        v = (float('inf'), None)
        successorindex = agentIndex + 1
        for action in gameState.getLegalActions(agentIndex):
            successor = gameState.generateSuccessor(agentIndex, action)
            v = min([v, (self.value(successor, successorindex, depth, alpha, beta)[0], action)], key = lambda x: x[0])
            if v[0] < alpha: 
                return v 
            beta = min(beta, v[0])
        agentIndex += 1
        return v 

    def maxvalue(self, gameState, agentIndex, depth, alpha, beta):
        v = (-float('inf'), None)
        successorindex = 1
        for action in gameState.getLegalActions(agentIndex):
            successor = gameState.generateSuccessor(agentIndex, action) 
            v = max([v, (self.value(successor, successorindex, depth, alpha, beta)[0], action)], key = lambda x: x[0])
            if v[0] > beta: 
                return v 
            alpha = max(alpha, v[0])
        return v 

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        agentIndex = self.index
        depth = 0
        alpha = -float('inf')
        beta = float('inf')
        return self.value(gameState, agentIndex, depth, alpha, beta)[1]
        

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """
    def value(self, gameState, agentIndex, depth):
        if agentIndex == (gameState.getNumAgents()): 
            agentIndex = 0
        if gameState.isLose():
            return ((self.evaluationFunction(gameState), None))
        if gameState.isWin():
            return ((self.evaluationFunction(gameState), None))
        if agentIndex == 0:
            depth += 1 
        if depth > self.depth: 
            return ((self.evaluationFunction(gameState), None))
        if agentIndex == 0:
            return self.maxvalue(gameState, agentIndex, depth)
                
        else:
            
            return self.expvalue(gameState, agentIndex, depth)

    def expvalue(self, gameState, agentIndex, depth):
        v = (0, None)
        successorindex = agentIndex + 1
        for action in gameState.getLegalActions(agentIndex):
            successor = gameState.generateSuccessor(agentIndex, action)
            p = 1/len(gameState.getLegalActions(agentIndex))
            v = (v[0] + p*self.value(successor, successorindex, depth)[0], action)
        agentIndex += 1
        return v 

    def maxvalue(self, gameState, agentIndex, depth):
        v = (-float('inf'), None)
        successorindex = 1
        for action in gameState.getLegalActions(agentIndex):
            successor = gameState.generateSuccessor(agentIndex, action) 
            v = max([v, (self.value(successor, successorindex, depth)[0], action)], key = lambda x: x[0])
        return v 
 


    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        agentIndex = self.index
        depth = 0 
        return self.value(gameState, agentIndex, depth)[1]

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    Pos = currentGameState.getPacmanPosition()
    Food = currentGameState.getFood()
    GhostStates = currentGameState.getGhostStates()
    ScaredTimes = [ghostState.scaredTimer for ghostState in GhostStates]
    GhostPos = currentGameState.getGhostPositions()
    Score = currentGameState.getScore()
    NumFood = currentGameState.getNumFood()
    Win = currentGameState.isWin()

    disghostlist = []
    for ghost in GhostPos:
        disghostlist += [manhattanDistance(Pos, ghost)]
    disghost = min(disghostlist)

    foodlist = [] 
    for food in Food.asList():
        foodlist += [manhattanDistance(Pos, food)]
    disfood = 0
    if len(foodlist) != 0:  
        disfood = min(foodlist)

    if sum(ScaredTimes) > 0: 
        disghost = float('inf')

    return 1/(disghost+1.5)*disghost - NumFood - 2.5/(disfood + 1) * disfood + Score 



# Abbreviation
better = betterEvaluationFunction
