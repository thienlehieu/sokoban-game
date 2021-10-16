import heapq, collections
from collections import deque


############################################################
# Abstract interfaces for search problems and search algorithms.

class SearchProblem:
    # Return the start state.
    def start(self): raise NotImplementedError("Override me")

    # Return whether |state| is an end state or not.
    def goalp(self, state): raise NotImplementedError("Override me")

    # Return a list of (action, newState, cost) tuples corresponding to edges
    # coming out of |state|.
    def expand(self, state): raise NotImplementedError("Override me")

class SearchAlgorithm:
    # - self.actions: list of actions that takes one from the start state to an end
    #                 state; if no action sequence exists, set it to None.
    # - self.totalCost: the sum of the costs along the path or None if no valid
    #                   action sequence exists.
    def solve(self, problem): raise NotImplementedError("Override me")

############################################################
# A* search algorithm
class AStarSearch(SearchAlgorithm):

    def __init__(self, heuristic):
        self.heuristic = heuristic

    def solve(self, problem):
        # If a path exists, set |actions| and |totalCost| accordingly.
        # Otherwise, leave them as None.
        self.actions = None
        self.totalCost = None
        self.numStatesExplored = 0
        self.finalCosts = collections.defaultdict(lambda:float('inf'))

        # Initialize data structures
        frontier = PriorityQueue()  # Explored states are maintained by the frontier.
        backpointers = {}  # map state to (action, previous state)

        # Add the start state
        startState = problem.start()
        self.finalCosts[startState] = 0
        frontier.update(startState, self.heuristic(startState))
        maxQSize = 0

        while True:
            # Remove the state from the queue with the lowest pastCost
            # (priority).
            state, estimatedCost = frontier.removeMin()
            if state == None: break

            pastCost = self.finalCosts[state]

            self.numStatesExplored += 1
            # Check if we've reached an end state; if so, extract solution.
            if problem.goalp(state):
                print("max queue length:", len(frontier.priorities))
                self.actions = []
                while state != startState:
                    action, prevState = backpointers[state]
                    self.actions.append(action)
                    state = prevState
                self.actions.reverse()
                self.totalCost = pastCost
                return

            # Expand from |state| to new successor states,
            # updating the frontier with each newState.
            for action, newState, cost in problem.expand(state):
                newPastCost = pastCost + cost
                self.finalCosts[newState] = min(newPastCost,self.finalCosts[newState])

                if frontier.update(newState, newPastCost + self.heuristic(newState)):
                    if len(frontier.priorities) > maxQSize: maxQSize = len(frontier.priorities)
                # Found better way to go to |newState|, update backpointer.
                    backpointers[newState] = (action, state)

# Breath first search algorithm
class bfs(SearchProblem):
    def solve(self, problem):
        self.actions = None
        self.totalCost = 0
        # initialize queue
        searchQueue = deque()
        self.numStatesExplored = 0
        backpointers = {}
        maxQSize = 1
        startState = problem.start()
        searchQueue.append(startState)
        while True:
            state = searchQueue.popleft() # pop new state from queue
            if state == None: break
            self.numStatesExplored += 1
            # check if goal state
            if problem.goalp(state):
                print("max queue length:", maxQSize)
                self.actions = []
                while state != startState:
                    action, prevState = backpointers[state]
                    self.actions.append(action)
                    state = prevState
                self.actions.reverse()
                return
            # explore new state
            for action, newState, cost in problem.expand(state):
                if newState not in backpointers: # f one state have already explored, then skip
                    searchQueue.append(newState)
                    if len(searchQueue) > maxQSize: maxQSize = len(searchQueue)
                    backpointers[newState] = (action, state)

# Data structure for supporting uniform cost search.
class PriorityQueue:
    def  __init__(self):
        self.DONE = -100000
        self.heap = []
        self.priorities = {}  # Map from state to priority

    # Insert |state| into the heap with priority |newPriority| if
    # |state| isn't in the heap or |newPriority| is smaller than the existing
    # priority.
    # Return whether the priority queue was updated.
    def update(self, state, newPriority):
        oldPriority = self.priorities.get(state)
        if oldPriority == None or newPriority < oldPriority:
            self.priorities[state] = newPriority
            heapq.heappush(self.heap, (newPriority, state))
            return True
        return False

    # Returns (state with minimum priority, priority)
    # or (None, None) if the priority queue is empty.
    def removeMin(self):
        while len(self.heap) > 0:
            priority, state = heapq.heappop(self.heap)
            if self.priorities[state] == self.DONE: continue  # Outdated priority, skip
            self.priorities[state] = self.DONE
            return (state, priority)
        return (None, None) # Nothing left...