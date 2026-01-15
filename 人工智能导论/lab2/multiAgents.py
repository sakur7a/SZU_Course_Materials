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
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
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

    def evaluationFunction(self, currentGameState: GameState, action):
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

        # Evaluation score
        score = successorGameState.getScore()


        # Get the positions of all foods in the new state
        foodList = newFood.asList()

        # Calculate the minimum distance to food 
        minFoodDistance = float('inf')
        if len(foodList) > 0:
            minFoodDistance = min([manhattanDistance(food, newPos) for food in foodList])
        else:
            minFoodDistance = 0

        # Prevents division by 0
        score += 1.0 / (minFoodDistance + 1)

        # Give a high score if eat food
        if currentGameState.getNumFood() > successorGameState.getNumFood():
            score += 200


        for ghostState in newGhostStates:
            if manhattanDistance(newPos, ghostState.getPosition()) <= 1:
                if ghostState.scaredTimer == 0:
                    score -= 1000
                else:
                    score += 1000

        # Give punishment if stop
        if action is Directions.STOP:
            score -= 10

        return score

def scoreEvaluationFunction(currentGameState: GameState):
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

    def getAction(self, gameState: GameState):
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
        
        pacmanAgentIndex = 0

        # Get all pacman's actions
        legalActions = gameState.getLegalActions(pacmanAgentIndex)

        # init
        bestAction = None
        maxValue = -float('inf')

        # Iterate all possible actions for Pacman
        for action in legalActions:
            successorState = gameState.generateSuccessor(pacmanAgentIndex, action)

            # 1 represent ghostAgentIndex
            value = self.minimax(successorState, 1, 0)

            # Update if we found a better move
            if value > maxValue:
                maxValue = value
                bestAction = action

        return bestAction

    def minimax(self, gameState: GameState, agentIndex: int, depth: int):
        """
        Recursive helper function to compute the minimax value of a state.
        """
        # Solve multiple ghosts
        numAgents = gameState.getNumAgents()

        # Base case: the current depth equals the max depth or the game is won or lose
        if depth == self.depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        
        # Get all current agent's legal actions
        legalActions = gameState.getLegalActions(agentIndex)

        if not legalActions:
            return self.evaluationFunction(gameState)
        
        nextAgentIndex = (agentIndex + 1) % numAgents
        nextDepth = depth + 1 if nextAgentIndex == 0 else depth

        # Generate values for all successor states
        successorValues = []
        for action in legalActions:
            successorState = gameState.generateSuccessor(agentIndex, action)
            successorValues.append(self.minimax(successorState, nextAgentIndex, nextDepth))

        # Pacman's turn
        if agentIndex == 0:
            return max(successorValues)
        # Ghost's turn
        else:
            return min(successorValues)
        

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """

        pacmanAgentIndex = 0
        alpha = -float('inf')
        beta = float('inf')

        # Get all pacman's actions
        legalActions = gameState.getLegalActions(pacmanAgentIndex)

        # init
        bestAction = None
        maxValue = -float('inf')

        # Iterate all possible actions for Pacman
        for action in legalActions:
            successorState = gameState.generateSuccessor(pacmanAgentIndex, action)

            # 1 represent ghostAgentIndex
            value = self.alphaBetaSearch(successorState, 1, 0, alpha, beta)

            # Update if we found a better move
            if value > maxValue:
                maxValue = value
                bestAction = action

            # Update alpha
            alpha = max(alpha, value)

        return bestAction
    

    def alphaBetaSearch(self, gameState: GameState, agentIndex: int, depth: int, alpha: int, beta: int):
        """
        Recursive helper function for alpha-beta search.
        """
        # Solve multiple ghosts
        numAgents = gameState.getNumAgents()

        # Base case: the current depth equals the max depth or the game is won or lose
        if depth == self.depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        
        # Get all current agent's legal actions
        legalActions = gameState.getLegalActions(agentIndex)

        if not legalActions:
            return self.evaluationFunction(gameState)
        
        # Max node
        if agentIndex == 0:
            value = -float('inf')
            for action in legalActions:
                successorState = gameState.generateSuccessor(agentIndex, action)
                value = max(value, self.alphaBetaSearch(successorState, 1, depth, alpha, beta))
            
                # Pruning
                if value > beta:
                    return value
                alpha = max(alpha, value)
            return value
        # Min node
        else:
            value = float('inf')
            nextAgentIndex = (agentIndex + 1) % numAgents
            # Determine next depth
            nextDepth = depth + 1 if nextAgentIndex == 0 else depth
            
            for action in legalActions:
                successorState = gameState.generateSuccessor(agentIndex, action)

                value = min(value, self.alphaBetaSearch(successorState, nextAgentIndex, nextDepth, alpha, beta))
                # Pruning
                if value < alpha:
                    return value
                beta = min(beta, value)
            return value

    

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        pacmanAgentIndex = 0
        
        # Get all legal actions for Pacman
        legalActions = gameState.getLegalActions(pacmanAgentIndex)
        
        bestAction = None
        maxValue = -float('inf')

        # Iterate through all possible actions for Pacman to find the one with the highest expectimax value
        for action in legalActions:
            successorState = gameState.generateSuccessor(pacmanAgentIndex, action)
            # The next agent is the first ghost (agent 1), initial depth is 0
            value = self.expectimax(successorState, 1, 0)
            
            if value > maxValue:
                maxValue = value
                bestAction = action
                
        return bestAction
    
    def expectimax(self, gameState: GameState, agentIndex: int, depth: int):
        """
        Recursive helper function for expectimax search.
        """
        numAgents = gameState.getNumAgents()

        # Base case: terminate if max depth is reached or game is over
        if depth == self.depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)

        legalActions = gameState.getLegalActions(agentIndex)
        if not legalActions:
            return self.evaluationFunction(gameState)

        # Determine the next agent and depth
        nextAgentIndex = (agentIndex + 1) % numAgents
        nextDepth = depth + 1 if nextAgentIndex == 0 else depth

        # Pacman's turn
        if agentIndex == 0:
            value = -float('inf')
            for action in legalActions:
                successorState = gameState.generateSuccessor(agentIndex, action)
                value = max(value, self.expectimax(successorState, nextAgentIndex, nextDepth))
            return value
        
        # Ghosts' turn
        else:
            totalValue = 0
            for action in legalActions:
                successorState = gameState.generateSuccessor(agentIndex, action)
                # Sum up the values from all possible ghost moves
                totalValue += self.expectimax(successorState, nextAgentIndex, nextDepth)

            return totalValue / len(legalActions)
    

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    
    currentFood = currentGameState.getFood()
    currentPos = currentGameState.getPacmanPosition()
    currentGhostStates = currentGameState.getGhostStates()
    curretnScaredTimes = [ghostState.scaredTimer for ghostState in currentGhostStates]
    currentCapsules = currentGameState.getCapsules()

    score = currentGameState.getScore()

    # Get the positions of all foods in the new state
    foodList = currentFood.asList()

    # Calculate the minimum distance to food 
    minFoodDistance = float('inf')
    if len(foodList) > 0:
        minFoodDistance = min([manhattanDistance(food, currentPos) for food in foodList])
    else:
        minFoodDistance = 0
    score += 1.0 / (minFoodDistance + 1)
    score -= 2 * len(foodList)
    score -= 20 * len(currentCapsules)

    for ghostState in currentGhostStates:
        distanceToGhost = manhattanDistance(currentPos, ghostState.getPosition())
        if ghostState.scaredTimer > 0:
            if distanceToGhost > 0:
                 score += 200 / distanceToGhost
        else:
            if distanceToGhost <= 1:
                score -= 1000
    return score

# Abbreviation
better = betterEvaluationFunction
