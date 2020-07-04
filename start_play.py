# -*- coding: utf-8 -*-
"""
human VS AI models
Input your move in the format: 2,3

@author: Junxiao Song and Yongjie Xu
"""

from __future__ import print_function
import pickle
from game import Board, Game
from mcts_pure import MCTSPlayer as MCTS_Pure
from mcts_alphaZero import MCTSPlayer
from policy_value_net_numpy import PolicyValueNetNumpy
# from policy_value_net import PolicyValueNet  # Theano and Lasagne
# from policy_value_net_pytorch import PolicyValueNet  # Pytorch
# from policy_value_net_tensorflow import PolicyValueNet # Tensorflow
# from policy_value_net_keras import PolicyValueNet  # Keras

def run():
    n = 5
    width, height = 8, 8
    model_file = 'best_policy_8_8_5.model'
    try:
        board = Board(width=width, height=height, n_in_row=n)
        game = Game(board)

        # ############### human VS AI ###################
        # load the trained policy_value_net in either Theano/Lasagne, PyTorch or TensorFlow

        # best_policy = PolicyValueNet(width, height, model_file = model_file)
        # mcts_player = MCTSPlayer(best_policy.policy_value_fn, c_puct=5, n_playout=400)

        # load the provided model (trained in Theano/Lasagne) into a MCTS player written in pure numpy
        try:
            policy_param = pickle.load(open(model_file, 'rb'))
        except:
            policy_param = pickle.load(open(model_file, 'rb'),
                                       encoding='bytes')  # To support python3
        best_policy = PolicyValueNetNumpy(width, height, policy_param)
        mcts_player = MCTSPlayer(best_policy.policy_value_fn,
                                 c_puct=5,
                                 n_playout=400)  # set larger n_playout for better performance


        #pure mcts player
        #make quick_play=True to enable a weaker but much faster roll-out player without mcts
        pure_mcts_player = MCTS_Pure(c_puct=1, n_playout=600, quick_play=False)
        roll_out_player = MCTS_Pure(quick_play=True)

        #1.run with two human player
        game.start_play_with_UI()

        #2.run with alpha zero nerutral network AI, and my quick roll-out AI
        #game.start_play_with_UI(AI=mcts_player, AI2 = roll_out_player)

        #3.run with alpha zero nerutral network AI, and my pure mcts AI
        #game.start_play_with_UI(AI=mcts_player, AI2 = pure_mcts_player)


    except KeyboardInterrupt:
        print('\n\rquit')


if __name__ == '__main__':
    run()
