"""
board.py
Cmput 455 sample code
Written by Cmput 455 TA and Martin Mueller

Implements a basic Go board with functions to:
- initialize to a given board size
- check if a move is legal
- play a move

The board uses a 1-dimensional representation with padding
"""

import numpy as np
import random
from typing import List, Tuple

from board_base import (
    board_array_size,
    coord_to_point,
    is_black_white,
    is_black_white_empty,
    opponent,
    where1d,
    BLACK,
    WHITE,
    EMPTY,
    BORDER,
    MAXSIZE,
    NO_POINT,
    PASS,
    GO_COLOR,
    GO_POINT,
)


"""
The GoBoard class implements a board and basic functions to play
moves, check the end of the game, and count the acore at the end.
The class also contains basic utility functions for writing a Go player.
For many more utility functions, see the GoBoardUtil class in board_util.py.

The board is stored as a one-dimensional array of GO_POINT in self.board.
See coord_to_point for explanations of the array encoding.
"""
class GoBoard(object):
    def __init__(self, size: int) -> None:
        """
        Creates a Go board of given size
        """
        assert 2 <= size <= MAXSIZE
        self.reset(size)
        self.calculate_rows_cols_diags()
        self.black_captures = 0
        self.white_captures = 0
        self.policy = "random"
        self.num_simulations = 10
        self.potential_winning_points = set()

    def simulate_move(self, move: GO_POINT)->float:
        move_color : GO_COLOR = self.current_player
        num_wins = 0
        for _ in range(self.num_simulations):
            temp_board = self.copy()
            temp_board.play_move(move,move_color)
            while temp_board.isEndOfGame() == False:
                empty_points = temp_board.get_empty_points()
                random.shuffle(empty_points)
                temp_board.play_move(empty_points[0],temp_board.current_player)
            
            if move_color == temp_board.get_winner():
                num_wins+=1
        
        return float(num_wins / self.num_simulations)
                
    def simulate(self) -> GO_POINT:
        """ Simulates for each empty point a random sequence of moves, and picks the move that yields the highest win rate"""
        empty_points = self.get_empty_points()
        best_move : GO_POINT = 0
        best_score = float("-inf")

        for point in empty_points:
            score = self.simulate_move(point)
            if score > best_score:
                best_score = score
                best_move = point
        
        return best_move

    def add_two_captures(self, color: GO_COLOR) -> None:
        if color == BLACK:
            self.black_captures += 2
        elif color == WHITE:
            self.white_captures += 2
    def get_captures(self, color: GO_COLOR) -> None:
        if color == BLACK:
            return self.black_captures
        elif color == WHITE:
            return self.white_captures

    def in_a_row_returns_list(self,point:GO_POINT,action:int,num_consecutive:list)->list:
        
        """
        Check how many of the point there are in a row, returns after 4 as that guarantees a win\n
        For action:\n
        \t0 means check north\n
        \t1 means check north-east\n
        \t2 means check east\n
        \t3 means check south-east\n
        \t4 means check south\n
        \t5 means check south-west\n
        \t6 means check west\n
        \t7 means check north-west\n
        """
        point_to_check = point
        if action == 0:
            point_to_check = point + self.NS
        elif action == 1:
            point_to_check = (point + self.NS) + 1
        elif action == 2:
            point_to_check = point + 1
        elif action == 3:
            point_to_check = (point - self.NS) + 1
        elif action == 4:
            point_to_check = point - self.NS
        elif action == 5:
            point_to_check = (point - self.NS) - 1
        elif action == 6:
            point_to_check = point - 1 
        elif action == 7:
            point_to_check = (point + self.NS) - 1

        try:
            state : GO_COLOR = self.board[point]
            if state == self.current_player:
                num_consecutive.append(point)
                if len(num_consecutive) == 4:
                    #meaning that there is enough to win the game, so return
                    return num_consecutive
                else:
                    return self.in_a_row_returns_list(point_to_check,action,num_consecutive)
            else:
                return num_consecutive
        except:
            return num_consecutive


    def in_a_row(self,point:GO_POINT,action:int,num_consecutive:int)->int:
        
        """
        Check how many of the point there are in a row, returns after 4 as that guarantees a win\n
        For action:\n
        \t0 means check north\n
        \t1 means check north-east\n
        \t2 means check east\n
        \t3 means check south-east\n
        \t4 means check south\n
        \t5 means check south-west\n
        \t6 means check west\n
        \t7 means check north-west\n
        """
        point_to_check = point
        if action == 0:
            point_to_check = point + self.NS
        elif action == 1:
            point_to_check = (point + self.NS) + 1
        elif action == 2:
            point_to_check = point + 1
        elif action == 3:
            point_to_check = (point - self.NS) + 1
        elif action == 4:
            point_to_check = point - self.NS
        elif action == 5:
            point_to_check = (point - self.NS) - 1
        elif action == 6:
            point_to_check = point - 1 
        elif action == 7:
            point_to_check = (point + self.NS) - 1

        try:
            state : GO_COLOR = self.board[point]
            if state == self.current_player:
                num_consecutive+=1
                if num_consecutive == 4:
                    #meaning that there is enough to win the game, so return
                    return num_consecutive
                else:
                    return self.in_a_row(point_to_check,action,num_consecutive)
            else:
                return num_consecutive
        except:
            return num_consecutive

    def in_a_row_until_empty_point(self,point:GO_POINT,action:int,num_consecutive:int)->int:
        
        """
        Check how many of the point there are in a row until an empty slot\n
        Different than in_a_row(), as that would count until the final point which could be anyting\n
        The last point here would have to be an empty point\n
        For action:\n
        \t0 means check north\n
        \t1 means check north-east\n
        \t2 means check east\n
        \t3 means check south-east\n
        \t4 means check south\n
        \t5 means check south-west\n
        \t6 means check west\n
        \t7 means check north-west\n
        """
        point_to_check = point
        if action == 0:
            point_to_check = point + self.NS
        elif action == 1:
            point_to_check = (point + self.NS) + 1
        elif action == 2:
            point_to_check = point + 1
        elif action == 3:
            point_to_check = (point - self.NS) + 1
        elif action == 4:
            point_to_check = point - self.NS
        elif action == 5:
            point_to_check = (point - self.NS) - 1
        elif action == 6:
            point_to_check = point - 1 
        elif action == 7:
            point_to_check = (point + self.NS) - 1

        try:
            state : GO_COLOR = self.board[point]
            if state == self.current_player:
                num_consecutive+=1
                return self.in_a_row_until_empty_point(point_to_check,action,num_consecutive)
            elif state == EMPTY:
                return num_consecutive
            else:
                return 0
        except:
            return 0

    def check_if_open_four(self,move:GO_POINT)->bool:
        """
        Check if this move would get a win for the current player\n
        Return 2 if a win or the number of connections in each connection using the following formula:\n
        This move should take precedent over everything else if it guarantees a win\n
        \t\t(NUM_IN_ROW_FOR_ONE_DIRECTION)/ 4 and then adding up for each direction and dividing by 8\n
        Logic is that a multi-pronged attack is better than on with just four in one direction as that would get blocked\n
        \t0 means check north\n
        \t1 means check north-east\n
        \t2 means check east\n
        \t3 means check south-east\n
        \t4 means check south\n
        \t5 means check south-west\n
        \t6 means check west\n
        \t7 means check north-west\n
        """
        action_neighbor_mapping = {
            '0': self.NS,
            '1' : self.NS + 1,
            '2' :  1,
            '3' : -self.NS + 1,
            '4' : -self.NS,
            '5' : -self.NS - 1,
            '6' : -1,
            '7' : self.NS - 1}

        directions : list = [('0','4'),('1','5'),('2','6'),('3','7')] # check in each of the four cross sections to see if the result is more than 5, meaning a win


        for direction1,direction2 in directions:
            count_one = self.in_a_row_until_empty_point(move + action_neighbor_mapping[direction1],int(direction1),0)
            count_two = self.in_a_row_until_empty_point(move + action_neighbor_mapping[direction2],int(direction2),0)
            if count_one + count_two >= 3:
                return True
        return False

    def return_five_in_row_set(self,move:GO_POINT)->set:
        """
        Check if this move would get a win for the current player\n
        Return 2 if a win or the number of connections in each connection using the following formula:\n
        This move should take precedent over everything else if it guarantees a win\n
        \t\t(NUM_IN_ROW_FOR_ONE_DIRECTION)/ 4 and then adding up for each direction and dividing by 8\n
        Logic is that a multi-pronged attack is better than on with just four in one direction as that would get blocked\n
        \t0 means check north\n
        \t1 means check north-east\n
        \t2 means check east\n
        \t3 means check south-east\n
        \t4 means check south\n
        \t5 means check south-west\n
        \t6 means check west\n
        \t7 means check north-west\n
        """
        action_neighbor_mapping = {
            '0': self.NS,
            '1' : self.NS + 1,
            '2' :  1,
            '3' : -self.NS + 1,
            '4' : -self.NS,
            '5' : -self.NS - 1,
            '6' : -1,
            '7' : self.NS - 1}

        directions : list = [('0','4'),('1','5'),('2','6'),('3','7')] # check in each of the four cross sections to see if the result is more than 5, meaning a win


        for direction1,direction2 in directions:
            list_one = self.in_a_row_returns_list(move + action_neighbor_mapping[direction1],int(direction1),[])
            list_two = self.in_a_row_returns_list(move + action_neighbor_mapping[direction2],int(direction2),[])
            if len(list_one) + len(list_two) >= 4:
                list_one.extend(list_two)
                return set(list_one)
        return set()    

    def check_if_five_in_row(self,move:GO_POINT)->bool:
        """
        Check if this move would get a win for the current player\n
        Return 2 if a win or the number of connections in each connection using the following formula:\n
        This move should take precedent over everything else if it guarantees a win\n
        \t\t(NUM_IN_ROW_FOR_ONE_DIRECTION)/ 4 and then adding up for each direction and dividing by 8\n
        Logic is that a multi-pronged attack is better than on with just four in one direction as that would get blocked\n
        \t0 means check north\n
        \t1 means check north-east\n
        \t2 means check east\n
        \t3 means check south-east\n
        \t4 means check south\n
        \t5 means check south-west\n
        \t6 means check west\n
        \t7 means check north-west\n
        """
        action_neighbor_mapping = {
            '0': self.NS,
            '1' : self.NS + 1,
            '2' :  1,
            '3' : -self.NS + 1,
            '4' : -self.NS,
            '5' : -self.NS - 1,
            '6' : -1,
            '7' : self.NS - 1}

        directions : list = [('0','4'),('1','5'),('2','6'),('3','7')] # check in each of the four cross sections to see if the result is more than 5, meaning a win


        for direction1,direction2 in directions:
            count_one = self.in_a_row(move + action_neighbor_mapping[direction1],int(direction1),0)
            count_two = self.in_a_row(move + action_neighbor_mapping[direction2],int(direction2),0)
            if count_one + count_two >= 4:
                return True
        return False
    
    def get_all_possible_wins(self)->set:
        """Combs through all empty points to see which ones at this moment would yield what potential winners"""
        empty_points = self.get_empty_points()
        return_set = set()
        for point in empty_points:
            is_win = self.check_if_five_in_row(point)
            if is_win:
                return_set.update(self.return_five_in_row_set(point))
        return return_set
    
    def blocks_win(self,move:GO_POINT):
        """
        Returns 1 if it blocks the enemy from winning
        Returns a number from 0 to 1 otherwise, telling us how strong the block would be
        """
        #for this, simply switch the self.current player and call get_win_or_connection_count, then swap it back
        
        self.current_player = opponent(self.current_player)

        result = self.check_win(move)
        self.current_player = opponent(self.current_player)
        if result:
            return True
        else:
            captures : set = self.get_captures_set(move)
            self.current_player = opponent(self.current_player)
            result = self.get_all_possible_wins()
            self.current_player = opponent(self.current_player)
            if len(captures.intersection(result)) > 0:
                return True
            else:
                return False
        
        

    def get_captures_count(self,move:GO_POINT) -> int:
        """
        Get the number of captures that would be made by making the move\n
        """

        action_neighbor_mapping = {
            '0': self.NS,
            '1' : self.NS + 1,
            '2' :  1,
            '3' : -self.NS + 1,
            '4' : -self.NS,
            '5' : -self.NS - 1,
            '6' : -1,
            '7' : self.NS - 1}
        num_captures = 0
        for action in action_neighbor_mapping:
            num_captures += self.niniku_capture(move + action_neighbor_mapping[action],int(action),0)
        
        return num_captures

    def get_captures_set(self,move:GO_POINT)->set:
        """
        Get the set of captures that would be made by making the move\n
        """
        return_list = []
        action_neighbor_mapping = {
            '0': self.NS,
            '1' : self.NS + 1,
            '2' :  1,
            '3' : -self.NS + 1,
            '4' : -self.NS,
            '5' : -self.NS - 1,
            '6' : -1,
            '7' : self.NS - 1}
        for action in action_neighbor_mapping:
            return_list.extend(self.niniku_capture_return_list(move + action_neighbor_mapping[action],int(action),[]))
        return_set = set(return_list)
        return return_set

    def niniku_capture_return_list(self,point:GO_POINT,action:int,captures:list)->list:
        """
        Recursively add to the list of stones taken\n
        if ran into a stone of the same color, then stop and return the list\n
        If ran into a boundary, return an empty list\n
        For action:\n
        \t0 means check north\n
        \t1 means check north-east\n
        \t2 means check east\n
        \t3 means check south-east\n
        \t4 means check south\n
        \t5 means check south-west\n
        \t6 means check west\n
        \t7 means check north-west\n
        """
        point_to_check = point
        if action == 0:
            point_to_check = point + self.NS
        elif action == 1:
            point_to_check = (point + self.NS) + 1
        elif action == 2:
            point_to_check = point + 1
        elif action == 3:
            point_to_check = (point - self.NS) + 1
        elif action == 4:
            point_to_check = point - self.NS
        elif action == 5:
            point_to_check = (point - self.NS) - 1
        elif action == 6:
            point_to_check = point - 1 
        elif action == 7:
            point_to_check = (point + self.NS) - 1

        try:
            state : GO_COLOR = self.board[point]
            if state == opponent(self.current_player):
                captures.append(point)
                return self.niniku_capture_return_list(point_to_check,action,captures)
            elif state == self.current_player:
                if len(captures) % 2 == 0:
                    return captures
                else:
                    return []
            elif state == EMPTY or state == BORDER:
                return []
        except:
            return []

    def niniku_capture(self,point:GO_POINT,action:int,num_captures:int)->int:
        """
        Recursively add to the list of stones taken\n
        if ran into a stone of the same color, then stop and return the list\n
        If ran into a boundary, return an empty list\n
        For action:\n
        \t0 means check north\n
        \t1 means check north-east\n
        \t2 means check east\n
        \t3 means check south-east\n
        \t4 means check south\n
        \t5 means check south-west\n
        \t6 means check west\n
        \t7 means check north-west\n
        """
        point_to_check = point
        if action == 0:
            point_to_check = point + self.NS
        elif action == 1:
            point_to_check = (point + self.NS) + 1
        elif action == 2:
            point_to_check = point + 1
        elif action == 3:
            point_to_check = (point - self.NS) + 1
        elif action == 4:
            point_to_check = point - self.NS
        elif action == 5:
            point_to_check = (point - self.NS) - 1
        elif action == 6:
            point_to_check = point - 1 
        elif action == 7:
            point_to_check = (point + self.NS) - 1

        try:
            state : GO_COLOR = self.board[point]
            if state == opponent(self.current_player):
                num_captures +=1
                return self.niniku_capture(point_to_check,action,num_captures)
            elif state == self.current_player:
                if num_captures % 2 == 0:
                    return num_captures
                else:
                    return 0
            elif state == EMPTY or state == BORDER:
                return 0
        except:
            return 0
        
    def check_win(self,move: GO_POINT)->bool:
        """Checks to see if move would result in a win for the move"""
        captures_count = self.get_captures_count(move)
        is_win = self.check_if_five_in_row(move)
        if self.current_player == WHITE:
            captures_count += self.white_captures
        elif self.current_player == BLACK:
            captures_count += self.black_captures

        if is_win or captures_count >= 10:
            return True
        else:
            return False
        
    def check_if_captures(self, move: GO_POINT)->bool:
        if self.get_captures_count(move) >= 2:
            return True
        else:
            return False
        
    def rule_based_silumation(self,verbose=False):
        """If verbose mode is True, False by default, return rule, and possible moves \n
        If False, return move
        """
        if verbose:
            return(self.rule_based_move())
        else:
            rule, result = self.rule_based_move()
            if rule == "":
                return self.simulate()
            else:
                return result[0]

    def rule_based_move(self)->"tuple[str,list[int]]":
        """Return the best rule used and the number of moves that meet that rule's requirements"""
        # It's important that each function returns a boolean true or false
        rules = [self.check_win,self.blocks_win,self.check_if_open_four,self.check_if_captures]
        rule_names = ["Win","BlockWin","OpenFour","Capture"]
        rule_results = [list()] * len(rules)
        empty_points = self.get_empty_points()

        for i,rule in enumerate(rules):
            if i >= 1 and len(rule_results[i - 1]) > 0:
                return (rule_names[i-1],rule_results[i - 1])
            else:
                for point in empty_points:
                    if point == 35:
                        print("here")
                    if rule(point):
                        rule_results[i].append(point)

        if rule_results[len(rule_results) - 1].__len__ == 0:
            return ("",[])
        else:
            return (rule_names[len(rule_results) - 1],rule_results[len(rule_results) - 1])
        
        

    def calculate_rows_cols_diags(self) -> None:
        if self.size < 5:
            return
        # precalculate all rows, cols, and diags for 5-in-a-row detection
        self.rows = []
        self.cols = []
        for i in range(1, self.size + 1):
            current_row = []
            start = self.row_start(i)
            for pt in range(start, start + self.size):
                current_row.append(pt)
            self.rows.append(current_row)
            
            start = self.row_start(1) + i - 1
            current_col = []
            for pt in range(start, self.row_start(self.size) + i, self.NS):
                current_col.append(pt)
            self.cols.append(current_col)
        
        self.diags = []
        # diag towards SE, starting from first row (1,1) moving right to (1,n)
        start = self.row_start(1)
        for i in range(start, start + self.size):
            diag_SE = []
            pt = i
            while self.get_color(pt) == EMPTY:
                diag_SE.append(pt)
                pt += self.NS + 1
            if len(diag_SE) >= 5:
                self.diags.append(diag_SE)
        # diag towards SE and NE, starting from (2,1) downwards to (n,1)
        for i in range(start + self.NS, self.row_start(self.size) + 1, self.NS):
            diag_SE = []
            diag_NE = []
            pt = i
            while self.get_color(pt) == EMPTY:
                diag_SE.append(pt)
                pt += self.NS + 1
            pt = i
            while self.get_color(pt) == EMPTY:
                diag_NE.append(pt)
                pt += -1 * self.NS + 1
            if len(diag_SE) >= 5:
                self.diags.append(diag_SE)
            if len(diag_NE) >= 5:
                self.diags.append(diag_NE)
        # diag towards NE, starting from (n,2) moving right to (n,n)
        start = self.row_start(self.size) + 1
        for i in range(start, start + self.size):
            diag_NE = []
            pt = i
            while self.get_color(pt) == EMPTY:
                diag_NE.append(pt)
                pt += -1 * self.NS + 1
            if len(diag_NE) >=5:
                self.diags.append(diag_NE)
        assert len(self.rows) == self.size
        assert len(self.cols) == self.size
        assert len(self.diags) == (2 * (self.size - 5) + 1) * 2

    def reset(self, size: int) -> None:
        """
        Creates a start state, an empty board with given size.
        """
        self.size: int = size
        self.NS: int = size + 1
        self.WE: int = 1
        self.ko_recapture: GO_POINT = NO_POINT
        self.last_move: GO_POINT = NO_POINT
        self.last2_move: GO_POINT = NO_POINT
        self.current_player: GO_COLOR = BLACK
        self.maxpoint: int = board_array_size(size)
        self.board: np.ndarray[GO_POINT] = np.full(self.maxpoint, BORDER, dtype=GO_POINT)
        self._initialize_empty_points(self.board)
        self.calculate_rows_cols_diags()
        self.black_captures = 0
        self.white_captures = 0

    def copy(self) -> 'GoBoard':
        b = GoBoard(self.size)
        assert b.NS == self.NS
        assert b.WE == self.WE
        b.ko_recapture = self.ko_recapture
        b.last_move = self.last_move
        b.last2_move = self.last2_move
        b.current_player = self.current_player
        assert b.maxpoint == self.maxpoint
        b.board = np.copy(self.board)
        return b

    def get_color(self, point: GO_POINT) -> GO_COLOR:
        return self.board[point]

    def pt(self, row: int, col: int) -> GO_POINT:
        return coord_to_point(row, col, self.size)

    def _is_legal_check_simple_cases(self, point: GO_POINT, color: GO_COLOR) -> bool:
        """
        Check the simple cases of illegal moves.
        Some "really bad" arguments will just trigger an assertion.
        If this function returns False: move is definitely illegal
        If this function returns True: still need to check more
        complicated cases such as suicide.
        """
        assert is_black_white(color)
        if point == PASS:
            return True
        # Could just return False for out-of-bounds, 
        # but it is better to know if this is called with an illegal point
        assert self.pt(1, 1) <= point <= self.pt(self.size, self.size)
        assert is_black_white_empty(self.board[point])
        if self.board[point] != EMPTY:
            return False
        if point == self.ko_recapture:
            return False
        return True

    def is_legal(self, point: GO_POINT, color: GO_COLOR) -> bool:
        """
        Check whether it is legal for color to play on point
        This method tries to play the move on a temporary copy of the board.
        This prevents the board from being modified by the move
        """
        if point == PASS:
            return True
        board_copy: GoBoard = self.copy()
        can_play_move = board_copy.play_move(point, color)
        return can_play_move

    def end_of_game(self) -> bool:
        return self.last_move == PASS \
           and self.last2_move == PASS
           
    def get_empty_points(self) -> np.ndarray:
        """
        Return:
            The empty points on the board
        """
        return where1d(self.board == EMPTY)

    def row_start(self, row: int) -> int:
        assert row >= 1
        assert row <= self.size
        return row * self.NS + 1

    def _initialize_empty_points(self, board_array: np.ndarray) -> None:
        """
        Fills points on the board with EMPTY
        Argument
        ---------
        board: numpy array, filled with BORDER
        """
        for row in range(1, self.size + 1):
            start: int = self.row_start(row)
            board_array[start : start + self.size] = EMPTY

    def is_eye(self, point: GO_POINT, color: GO_COLOR) -> bool:
        """
        Check if point is a simple eye for color
        """
        if not self._is_surrounded(point, color):
            return False
        # Eye-like shape. Check diagonals to detect false eye
        opp_color = opponent(color)
        false_count = 0
        at_edge = 0
        for d in self._diag_neighbors(point):
            if self.board[d] == BORDER:
                at_edge = 1
            elif self.board[d] == opp_color:
                false_count += 1
        return false_count <= 1 - at_edge  # 0 at edge, 1 in center

    def _is_surrounded(self, point: GO_POINT, color: GO_COLOR) -> bool:
        """
        check whether empty point is surrounded by stones of color
        (or BORDER) neighbors
        """
        for nb in self._neighbors(point):
            nb_color = self.board[nb]
            if nb_color != BORDER and nb_color != color:
                return False
        return True

    def _has_liberty(self, block: np.ndarray) -> bool:
        """
        Check if the given block has any liberty.
        block is a numpy boolean array
        """
        for stone in where1d(block):
            empty_nbs = self.neighbors_of_color(stone, EMPTY)
            if empty_nbs:
                return True
        return False

    def _block_of(self, stone: GO_POINT) -> np.ndarray:
        """
        Find the block of given stone
        Returns a board of boolean markers which are set for
        all the points in the block 
        """
        color: GO_COLOR = self.get_color(stone)
        assert is_black_white(color)
        return self.connected_component(stone)

    def connected_component(self, point: GO_POINT) -> np.ndarray:
        """
        Find the connected component of the given point.
        """
        marker = np.full(self.maxpoint, False, dtype=np.bool_)
        pointstack = [point]
        color: GO_COLOR = self.get_color(point)
        assert is_black_white_empty(color)
        marker[point] = True
        while pointstack:
            p = pointstack.pop()
            neighbors = self.neighbors_of_color(p, color)
            for nb in neighbors:
                if not marker[nb]:
                    marker[nb] = True
                    pointstack.append(nb)
        return marker

    def _detect_and_process_capture(self, nb_point: GO_POINT) -> GO_POINT:
        """
        Check whether opponent block on nb_point is captured.
        If yes, remove the stones.
        Returns the stone if only a single stone was captured,
        and returns NO_POINT otherwise.
        This result is used in play_move to check for possible ko
        """
        single_capture: GO_POINT = NO_POINT
        opp_block = self._block_of(nb_point)
        if not self._has_liberty(opp_block):
            captures = list(where1d(opp_block))
            self.board[captures] = EMPTY
            if len(captures) == 1:
                single_capture = nb_point
        return single_capture
    
    def play_move(self, point: GO_POINT, color: GO_COLOR) -> bool:
        """
        Tries to play a move of color on the point.
        Returns whether or not the point was empty.
        """
        if self.board[point] != EMPTY:
            return False
        self.board[point] = color
        self.current_player = opponent(color)
        self.last2_move = self.last_move
        self.last_move = point
        O = opponent(color)
        offsets = [1, -1, self.NS, -self.NS, self.NS+1, -(self.NS+1), self.NS-1, -self.NS+1]
        for offset in offsets:
            if self.board[point+offset] == O and self.board[point+(offset*2)] == O and self.board[point+(offset*3)] == color:
                self.board[point+offset] = EMPTY
                self.board[point+(offset*2)] = EMPTY
                if color == BLACK:
                    self.black_captures += 2
                else:
                    self.white_captures += 2
        return True
    
    def isEndOfGame(self)->bool:
        if self.black_captures == 10 or self.white_captures == 10:
            return True
        elif self.detect_five_in_a_row() != EMPTY:
            return True
        elif len(self.get_empty_points()) == 0:
            return True
        else:
            return False
        
    def get_winner(self)-> GO_COLOR:
        if not self.isEndOfGame():
            return EMPTY
        else:
            if self.black_captures == 10:
                return BLACK
            elif self.white_captures == 10:
                return WHITE
            else:
                return self.detect_five_in_a_row()

    
    def neighbors_of_color(self, point: GO_POINT, color: GO_COLOR) -> List:
        """ List of neighbors of point of given color """
        nbc: List[GO_POINT] = []
        for nb in self._neighbors(point):
            if self.get_color(nb) == color:
                nbc.append(nb)
        return nbc

    def _neighbors(self, point: GO_POINT) -> List:
        """ List of all four neighbors of the point """
        return [point - 1, point + 1, point - self.NS, point + self.NS]

    def _diag_neighbors(self, point: GO_POINT) -> List:
        """ List of all four diagonal neighbors of point """
        return [point - self.NS - 1,
                point - self.NS + 1,
                point + self.NS - 1,
                point + self.NS + 1]

    def last_board_moves(self) -> List:
        """
        Get the list of last_move and second last move.
        Only include moves on the board (not NO_POINT, not PASS).
        """
        board_moves: List[GO_POINT] = []
        if self.last_move != NO_POINT and self.last_move != PASS:
            board_moves.append(self.last_move)
        if self.last2_move != NO_POINT and self.last2_move != PASS:
            board_moves.append(self.last2_move)
        return board_moves

    

    def detect_five_in_a_row(self) -> GO_COLOR:
        """
        Returns BLACK or WHITE if any five in a row is detected for the color
        EMPTY otherwise.
        """
        for r in self.rows:
            result = self.has_five_in_list(r)
            if result != EMPTY:
                return result
        for c in self.cols:
            result = self.has_five_in_list(c)
            if result != EMPTY:
                return result
        for d in self.diags:
            result = self.has_five_in_list(d)
            if result != EMPTY:
                return result
        return EMPTY

    def has_five_in_list(self, list) -> GO_COLOR:
        """
        Returns BLACK or WHITE if any five in a rows exist in the list.
        EMPTY otherwise.
        """
        prev = BORDER
        counter = 1
        for stone in list:
            if self.get_color(stone) == prev:
                counter += 1
            else:
                counter = 1
                prev = self.get_color(stone)
            if counter == 5 and prev != EMPTY:
                return prev
        return EMPTY