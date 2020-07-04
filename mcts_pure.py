# -*- coding: utf-8 -*-
"""
A pure implementation of the Monte Carlo Tree Search (MCTS)

@author: Junxiao Song
"""
import random, math
import numpy as np
import copy
from operator import itemgetter


def rollout_policy_fn(board):
    """a coarse, fast version of policy_fn used in the rollout phase."""
    # rollout randomly
    #action_probs = np.random.rand(len(board.availables))
    #return zip(board.availables, action_probs)
    return list(policy_value_fn(board)[0])

def policy_value_fn(board):
    """a function that takes in a state and outputs a list of (action, probability)
    tuples and a score for the state"""
    # return uniform probabilities and 0 score for pure MCTS
    r = [0,0,0,0]
    score = {}
    total_r = 0
    for point in board.availables:
        for k,d in enumerate((0, 1, 10, -10)):
            length, end = board.get_length(point, d, board.current_player)
            length1, end1 = board.get_length(point, d, 3-board.current_player)
            r[k] = max((end!=-1 or length>4)*length**2 + (end1!=-1 or length1>4)*length1**2 + end + end1, 1)
            if length>4:
                r[k] += 3000
            if length1>4:
                r[k] += 1800
            if (length==4 and end==1) or (length1==4 and end1==1):
                r[k] += 1000
        sum_r = sum(r)
        total_r += sum_r
        score[point] = sum_r
    for point in score:
        score[point]/=total_r
    return score.items(), 0


class TreeNode(object):
    """A node in the MCTS tree. Each node keeps track of its own value Q,
    prior probability P, and its visit-count-adjusted prior score u.
    """

    def __init__(self, parent, prior_p):
        self._parent = parent
        self._children = {}  # a map from action to TreeNode
        self._actions = {}
        self._n_visits = 0
        self._Q = 0
        self._u = 0
        self._P = prior_p

    def expand(self, state):
        """Expand tree by creating new children.
        action_priors: a list of tuples of actions and their prior probability
            according to the policy function.
        """
        items = set(self._actions.keys()) - set(self._children.keys())
        _ = np.array(list(items))
        pick = _[random.choice(list(np.where(_==max(_))[0]))]
        self._children[pick] = TreeNode(self, self._actions[pick])
        state.do_move(pick)
        return self._children[pick]

    def select(self, c_puct):
        """Select action among children that gives maximum action value Q
        plus bonus u(P).
        Return: A tuple of (action, next_node)
        """
        items = tuple(self._children.items())
        _ = np.array([__[1].get_value(c_puct) for __ in items])
        return items[random.choice(list(np.where(_==max(_))[0]))]

    def update(self, leaf_value):
        """Update node values from leaf evaluation.
        leaf_value: the value of subtree evaluation from the current player's
            perspective.
        """
        # Count visit.
        self._n_visits += 1
        # Update Q, a running average of values for all visits.
        self._Q += 1.0*(leaf_value - self._Q) / self._n_visits

    def update_recursive(self, leaf_value):
        """Like a call to update(), but applied recursively for all ancestors.
        """
        # If it is not root, this node's parent should be updated first.
        if self._parent:
            self._parent.update_recursive(1-leaf_value)
        self.update(leaf_value)

    def get_value(self, c_puct):
        """Calculate and return the value for this node.
        It is a combination of leaf evaluations Q, and this node's prior
        adjusted for its visit count, u.
        c_puct: a number in (0, inf) controlling the relative impact of
            value Q, and prior probability P, on this node's score.
        """
        self._u = (c_puct * self._P * 
          np.sqrt(math.log(self._parent._n_visits) / (1 + self._n_visits)))
        return self._Q + self._u

    def has_expanded(self):
        """Check if leaf node (i.e. no nodes below this have been expanded).
        """
        return len(self._actions)!=len(self._children) or not self._children

    def is_root(self):
        return self._parent is None


class MCTS(object):
    """A simple implementation of Monte Carlo Tree Search."""

    def __init__(self, policy_value_fn, c_puct=5, n_playout=10000):
        """
        policy_value_fn: a function that takes in a board state and outputs
            a list of (action, probability) tuples and also a score in [-1, 1]
            (i.e. the expected value of the end game score from the current
            player's perspective) for the current player.
        c_puct: a number in (0, inf) that controls how quickly exploration
            converges to the maximum-value policy. A higher value means
            relying on the prior more.
        """
        self._root = TreeNode(None, 1.0)
        self._policy = policy_value_fn
        self._c_puct = c_puct
        self._n_playout = n_playout

    def _playout(self, state):
        """Run a single playout from the root to the leaf, getting a value at
        the leaf and propagating it back through its parents.
        State is modified in-place, so a copy must be provided.
        """
        node = self._root
        while(1):
            if node.has_expanded() :
                break
            # Greedily select next move.
            action, node = node.select(self._c_puct)
            state.do_move(action)
        if not node._children:#is leaf
            action_probs, _ = self._policy(state)
            for a,p in action_probs:
                node._actions[a] = p
        # Check for end of game
        end, winner = state.game_end()
        if not end:
            node = node.expand(state)
        # Evaluate the leaf node by random rollout
        leaf_value = self._evaluate_rollout(state)
        # Update value and visit count of nodes in this traversal.
        node.update_recursive(1-leaf_value)

    def _evaluate_rollout(self, state, limit=1000):
        """Use the rollout policy to play until the end of the game,
        returning +1 if the current player wins, -1 if the opponent wins,
        and 0 if it is a tie.
        """
        player = state.get_current_player()
        for i in range(limit):
            end, winner = state.game_end()
            if end:
                break
            tmp = list(rollout_policy_fn(state))
            random.shuffle(tmp)
            max_action = max(tmp, key=lambda a:a[1])[0]
            '''
            action_probs = list(rollout_policy_fn(state))
            action_probs.sort(key=lambda a: a[1])
            actions = [_[0] for _ in action_probs[-2:]]
            probs = [_[1] for _ in action_probs[-2:]]
            '''
            '''
            _ = np.array([__[1] for __ in action_probs])
            max_action =  action_probs[random.choice(list(np.where(_==max(_))[0]))][0]
            '''
            #action = random.choices(actions, probs)[0]
            state.do_move(max_action)
        else:
            # If no break from the loop, issue a warning.
            print("WARNING: rollout reached move limit")
        if winner == -1:  # tie
            return 0.5
        else:
            return 1 if winner == player else 0

    def get_move(self, state):
        """Runs all playouts sequentially and returns the most visited action.
        state: the current game state

        Return: the selected action
        """
        for n in range(self._n_playout):
            state_copy = copy.deepcopy(state)
            self._playout(state_copy)
        items = tuple(self._root._children.items())
        _ = np.array([__[1]._n_visits for __ in items])
        return items[random.choice(list(np.where(_==max(_))[0]))][0]

    def update_with_move(self, last_move):
        """Step forward in the tree, keeping everything we already know
        about the subtree.
        """
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:
            self._root = TreeNode(None, 1.0)

    def __str__(self):
        return "MCTS"


class MCTSPlayer(object):
    """AI player based on MCTS"""
    def __init__(self, c_puct=5, n_playout=2000, quick_play=False):
        self.mcts = MCTS(policy_value_fn, c_puct, n_playout)
        self.quick_play = quick_play

    def set_player_ind(self, p):
        self.player = p

    def reset_player(self):
        self.mcts.update_with_move(-1)

    def get_action(self, board):
        import time
        sensible_moves = board.availables
        if len(sensible_moves) > 0:
            if self.quick_play:
                tmp = list(rollout_policy_fn(board))
                random.shuffle(tmp)
                move = max(tmp, key=lambda a:a[1])[0]
            else:
                self.mcts.update_with_move(board.last_move)
                move = self.mcts.get_move(board)
                self.mcts.update_with_move(move)
            #print(move)
            #time.sleep(0.5)
            return move
        else:
            print("WARNING: the board is full")

    def __str__(self):
        return "MCTS {}".format(self.player)
