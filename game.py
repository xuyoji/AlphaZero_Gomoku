# -*- coding: utf-8 -*-
"""
@author: Junxiao Song & xuyj & okarev-TT-33
"""

from __future__ import print_function
import numpy as np
from GUI_v1_4 import GUI

class Board(object):
    """board for the game"""

    def __init__(self, **kwargs):
        self.width = int(kwargs.get('width', 8))
        self.height = int(kwargs.get('height', 8))
        # board states stored as a dict,
        # key: move as location on the board,
        # value: player as pieces type
        self.states = {}
        # need how many pieces in a row to win
        self.n_in_row = int(kwargs.get('n_in_row', 5))
        self.players = [1, 2]  # player1 and player2

    def init_board(self, start_player=0):
        if self.width < self.n_in_row or self.height < self.n_in_row:
            raise Exception('board width and height can not be '
                            'less than {}'.format(self.n_in_row))
        self.start_player = self.players[start_player]
        self.current_player = self.players[start_player]  # start player
        # keep available moves in a list
        self.availables = set(range(self.width * self.height))
        self.availables_backup = set(range(self.width * self.height))
        self.states = {}
        self.last_move = -1
        self.forbid = set({})

    def move_to_location(self, move):
        """
        3*3 board's moves like:
        6 7 8
        3 4 5
        0 1 2
        and move 5's location is (1,2)
        """
        h = move // self.width
        w = move % self.width
        return [h, w]

    def location_to_move(self, location):
        if len(location) != 2:
            return -1
        h = location[0]
        w = location[1]
        move = h * self.width + w
        if move not in range(self.width * self.height):
            return -1
        return move

    def current_state(self):
        """return the board state from the perspective of the current player.
        state shape: 4*width*height
        """

        square_state = np.zeros((4, self.width, self.height))
        if self.states:
            moves, players = np.array(list(zip(*self.states.items())))
            move_curr = moves[players == self.current_player]
            move_oppo = moves[players != self.current_player]
            square_state[0][move_curr // self.width,
                            move_curr % self.height] = 1.0
            square_state[1][move_oppo // self.width,
                            move_oppo % self.height] = 1.0
            # indicate the last move location
            square_state[2][self.last_move // self.width,
                            self.last_move % self.height] = 1.0
        if len(self.states) % 2 == 0:
            square_state[3][:, :] = 1.0  # indicate the colour to play
        return square_state[:, ::-1, :]

    '''
    def has_line(self, point, length, direction):
        #check if there's a line with len(line) >= length 
        #in the axis of with orgin = point
        #axis: 0->x, 1->y, 10->(45 degree), -10->(-45 degree)
        #length can be negative
        end_free = False
        str_direction = str(direction)
        x, y = point%self.width, point//self.width
        def forward(x, y, step):
            if '0' in str_direction:
                x += step
            if '-1' in str_direction:
                y -= step
            elif '1' in str_direction:
                y += step
            return x, y, y*self.width+x
        x_end, y_end, p = forward(x, y, length)
        if not 0<=x_end<self.width or not 0<=y_end<self.height:
            return False
        for i in range(abs(length)):
            x, y, p = forward(x, y, 1)
            if self.states[p] != 1:
                return False
        return True
    '''

    def get_length(self, point, axis, player):
        #get the length of linked black on axis
        #after set black on the point 
        #axis: 0->x, 1->y, 10->(45 degree), -10->(-45 degree)
        #return length of linked black and end on the axis
        #end = -1 if both end with white or out of border
        #end = 0 if one is white or out of border and the other is empty
        #end = 1 if both are empty
        str_axis = str(axis)
        end = 1
        x, y = point%self.width, point//self.width
        dx, dy = 0, 0
        if '0' in str_axis:
            dx += 1
        if '-1' in str_axis:
            dy -= 1
        elif '1' in str_axis:
            dy += 1
        length = 1
        for k in (-1, 1):
            xc, yc = x, y
            while True:
                xc += k*dx
                yc += k*dy
                pc = yc*self.width+xc
                state = self.states.get(pc)
                if not (0<=xc<self.width and 0<=yc<self.height):
                    end -= 1
                    break
                elif state!=player:
                    if state:
                       end -= 1 
                    break
                elif state==player:
                    length += 1
        return length, end

    def check_forbid(self, point): 
        line_list = []
        for d in (0, 1, 10, -10):
            length, end = self.get_length(point, d, self.start_player)
            if length == 5:#win
                return False
            if length >5:#long link
                return True
            if length >2 and end >-1:
                if not (length==3 and end==0):
                    line_list.append((length, end))
        if len(line_list)<2:
            return False
        if len(line_list) == 2 and line_list[0][0]!=line_list[1][0]:#43
            return False
        return True

    def do_move(self, move):
        self.states[move] = self.current_player
        self.availables.remove(move)
        self.availables_backup.remove(move)
        if self.current_player != self.start_player:
            self.forbid = set({})
            for p in self.availables:
                if self.check_forbid(p):
                    self.forbid.add(p)
            self.availables = self.availables_backup-self.forbid
        else:
            self.availables = {_ for _ in self.availables_backup}
        self.current_player = (
            self.players[0] if self.current_player == self.players[1]
            else self.players[1]
        )
        self.last_move = move


    def has_a_winner(self):
        if self.last_move >=0:
            player = self.states[self.last_move]
            for d in (0, 1, 10, -10):
                if self.get_length(self.last_move, d, player)[0] >= 5:
                    return True,  player
        return False, -1

    def game_end(self):
        """Check whether the game is ended or not"""
        win, winner = self.has_a_winner()
        if win:
            return True, winner
        elif not len(self.availables):
            return True, -1
        return False, -1

    def get_current_player(self):
        return self.current_player


class Game(object):
    """game server"""

    def __init__(self, board, **kwargs):
        self.board = board

    def graphic(self, board, player1, player2):
        """Draw the board and show game info"""
        width = board.width
        height = board.height

        print("Player", player1, "with X".rjust(3))
        print("Player", player2, "with O".rjust(3))
        print()
        for x in range(width):
            print("{0:8}".format(x), end='')
        print('\r\n')
        for i in range(height - 1, -1, -1):
            print("{0:4d}".format(i), end='')
            for j in range(width):
                loc = i * width + j
                p = board.states.get(loc, -1)
                if loc == self.board.last_move:
                    if p == player1:
                        print('#'.center(8), end='')
                    elif p== player2:
                        print('@'.center(8), end='')
                elif p == player1:
                    print('X'.center(8), end='')
                elif p == player2:
                    print('O'.center(8), end='')
                else:
                    print('_'.center(8), end='')
            print('\r\n\r\n')

    def start_play_with_UI(self, AI=None, AI2=None, start_player=1):
        '''
        a GUI for playing
        '''
        if AI:
            AI.reset_player()
        self.board.init_board()
        current_player = SP = start_player
        UI = GUI(self.board.width)
        end = False
        while True:
            print('current_player', current_player)

            if current_player == 0:
                UI.show_messages('white turn')
            else:
                UI.show_messages('black turn')

            if AI and current_player == 1 and not end:
                move = AI.get_action(self.board)
            else:
                if AI2 and not end:
                    move = AI2.get_action(self.board)
                else:
                    inp = UI.get_input()
                    if not AI2 and inp[0] == 'move' and not end:
                        if type(inp[1]) != int:
                            move = UI.loc_2_move(inp[1])
                        else:
                            move = inp[1]
                    elif inp[0] == 'RestartGame':
                        end = False
                        current_player = SP
                        self.board.init_board()
                        UI.restart_game()
                        if AI:
                            AI.reset_player()
                        if AI2:
                            AI2.reset_player()
                        continue
                    elif inp[0] == 'ResetScore':
                        UI.reset_score()
                        continue
                    elif inp[0] == 'quit':
                        exit()
                        continue
                    elif inp[0] == 'SwitchPlayer':
                        end = False
                        self.board.init_board()
                        UI.restart_game(False)
                        UI.reset_score()
                        if AI:
                            AI.reset_player()
                        if AI2:
                            AI2.reset_player()
                        SP = (SP+1) % 2
                        current_player = SP
                        continue
                    else:
                        # print('ignored inp:', inp)
                        continue
            # print('player %r move : %r'%(current_player,[move//self.board.width,move%self.board.width]))
            if not end:
                # print(move, type(move), current_player)
                if move in self.board.availables:
                    UI.render_step(move, self.board.current_player)
                    self.board.do_move(move)
                    print('move', move%self.board.width, move//self.board.width, '\n')
                    # print(2, self.board.get_current_player())
                    current_player = (current_player + 1) % 2
                    # UI.render_step(move, current_player)
                    end, winner = self.board.game_end()
                    if end:
                        if winner != -1:
                            print("Game end. Winner is player", winner)
                            UI.add_score(winner)
                        else:
                            print("Game end. Tie")
                        print(UI.score)
                        print()
                else:
                    print('forbid hand!')

    def start_play(self, player1, player2, start_player=0, is_shown=1):
        """start a game between two players"""
        start_player_idx = start_player
        if start_player_idx not in (0, 1):
            raise Exception('start_player should be either 0 (player1 first) '
                            'or 1 (player2 first)')
        self.board.init_board(start_player_idx)
        p1, p2 = self.board.players
        player1.set_player_ind(p1)
        player2.set_player_ind(p2)
        players = {p1: player1, p2: player2}
        if is_shown:
            self.graphic(self.board, player1.player, player2.player)
        while True:
            current_player = self.board.get_current_player()
            player_in_turn = players[current_player]
            move = player_in_turn.get_action(self.board)
            self.board.do_move(move)
            if is_shown:
                self.graphic(self.board, player1.player, player2.player)
            end, winner = self.board.game_end()
            if end:
                if is_shown:
                    if winner != -1:
                        print("Game end. Winner is", players[winner])
                    else:
                        print("Game end. Tie")
                return winner

    def start_self_play(self, player, is_shown=0, temp=1e-3):
        """ start a self-play game using a MCTS player, reuse the search tree,
        and store the self-play data: (state, mcts_probs, z) for training
        """
        self.board.init_board()
        p1, p2 = self.board.players
        states, mcts_probs, current_players = [], [], []
        while True:
            move, move_probs = player.get_action(self.board,
                                                 temp=temp,
                                                 return_prob=1)
            # store the data
            states.append(self.board.current_state())
            mcts_probs.append(move_probs)
            current_players.append(self.board.current_player)
            # perform a move
            self.board.do_move(move)
            if is_shown:
                self.graphic(self.board, p1, p2)
            end, winner = self.board.game_end()
            if end:
                # winner from the perspective of the current player of each state
                winners_z = np.zeros(len(current_players))
                if winner != -1:
                    winners_z[np.array(current_players) == winner] = 1.0
                    winners_z[np.array(current_players) != winner] = -1.0
                # reset MCTS root node
                player.reset_player()
                if is_shown:
                    if winner != -1:
                        print("Game end. Winner is player:", winner)
                    else:
                        print("Game end. Tie")
                return winner, zip(states, mcts_probs, winners_z)
