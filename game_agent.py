"""This file contains all the classes you must complete for this project.

You can use the test cases in agent_test.py to help during development, and
augment the test suite with your own test cases to further test your code.

You must test your agent's strength against a set of agents with known
relative strength using tournament.py and include the results in your report.
"""
import random
import math
import sys
import numpy as np


class Timeout(Exception):
    """Subclass base exception for code clarity."""
    pass
def is_valid_move(move, MAX_ROW, MAX_COL):
    """ Check if move is out of the board size
    Args:
        move (tuple): (row, col)
        MAX_ROW (int): The length of vertical axis of the board size
        MAX_COL (int): The length of horizontal axis of the board size
    Returns:
        True or False
    """
    r, c = move
    if r < 0 or r >= MAX_ROW:
        return False
    if c < 0 or c >= MAX_COL:
        return False
    return True


def hueristic(row, col, MAX_VAL=10, discount=0.9, directions=[(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)]):
    """Returns custom value matrix, value
    value[row][col] implies how good move (row, col) is
    Args:
        row (int): Row size of the board
        col (int): Col size of the board
        MAX_VAL (int): The highest value (usually center of the board)
        discount (float): Starting from the best value, nearest cells will receive discounted value
        directions (list): Possible legal moves
    Returns:
        value (2 by 2 matrix): value[row][col] = some value
    """
    center_point = (row // 2, col // 2)
    value = [[0.0 for c in range(col)] for r in range(row)]
    change = True
    while change:
        change = False
        for r in range(len(value)):
            for c in range(len(value[0])):
                if (r, c) == center_point:
                    if value[r][c] != MAX_VAL:
                        value[r][c] = MAX_VAL
                        change = True
                else:
                    near_points = [(r + delta_r, c + delta_c) for delta_r,
                                   delta_c in directions if is_valid_move((r + delta_r, c + delta_c), row, col)]
                    max_near = max(value[x][y] for x, y in near_points)
                    if max_near * discount != value[r][c]:
                        value[r][c] = max_near * discount
                        change = True

    return value

# Define the VALUE here such that it doesn't have to compute multiple times
VALUE = hueristic(row=7, col=7, MAX_VAL=10, discount=0.9)

def hueristic_fun_1(game, player):
    # TODO: finish this function!
    # raise NotImplementedError
    global VALUE

    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    all_blank_spaces = game.get_blank_spaces()
    row, col = game.height, game.width

    opponent = game.get_opponent(player)
    my_legal_moves = game.get_legal_moves(player=player)
    opponent_legal_moves = game.get_legal_moves(player=opponent)

    my_score = len(my_legal_moves)
    opp_score = len(opponent_legal_moves)
    if len(my_legal_moves) > 0:
        my_score += np.mean([VALUE[r][c] for r, c in my_legal_moves])
    if len(opponent_legal_moves) > 0:
        opp_score += np.mean([VALUE[r][c] for r, c in opponent_legal_moves])

    return my_score - opp_score

def hueristic_fun_2(game, player):
    # TODO: finish this function!
    # raise NotImplementedError
    global VALUE

    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    all_blank_spaces = game.get_blank_spaces()
    row, col = game.height, game.width

    opponent = game.get_opponent(player)
    my_legal_moves = game.get_legal_moves(player=player)
    opponent_legal_moves = game.get_legal_moves(player=opponent)

    my_score = len(my_legal_moves)
    opp_score = len(opponent_legal_moves)
    if len(my_legal_moves) > 0:
        my_score += np.mean([VALUE[r][c] for r, c in my_legal_moves])
    if len(opponent_legal_moves) > 0:
        opp_score += np.mean([VALUE[r][c] for r, c in opponent_legal_moves])

    return my_score + 0.5*opp_score

def hueristic_fun_3(game, player):
    # TODO: finish this function!
    # raise NotImplementedError
    global VALUE

    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    all_blank_spaces = game.get_blank_spaces()
    row, col = game.height, game.width

    opponent = game.get_opponent(player)
    my_legal_moves = game.get_legal_moves(player=player)
    opponent_legal_moves = game.get_legal_moves(player=opponent)

    my_score = len(my_legal_moves)
    opp_score = len(opponent_legal_moves)
    if len(my_legal_moves) > 0:
        my_score += np.mean([VALUE[r][c] for r, c in my_legal_moves])
    if len(opponent_legal_moves) > 0:
        opp_score += np.mean([VALUE[r][c] for r, c in opponent_legal_moves])

    return my_score - 2/3*opp_score

class CustomPlayer:
    """Game-playing agent that chooses a move using your evaluation function
    and a depth-limited minimax algorithm with alpha-beta pruning. You must
    finish and test this player to make sure it properly uses minimax and
    alpha-beta to return a good move before the search time limit expires.

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    iterative : boolean (optional)
        Flag indicating whether to perform fixed-depth search (False) or
        iterative deepening search (True).

    method : {'minimax', 'alphabeta'} (optional)
        The name of the search method to use in get_move().

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """

    def __init__(self, search_depth=3, score_fn=hueristic_fun_1,
                 iterative=True, method='minimax', timeout=10.):
        self.search_depth = search_depth
        self.iterative = iterative
        self.score = score_fn
        self.method = method
        self.time_left = None
        self.TIMER_THRESHOLD = timeout

    def get_move(self, game, legal_moves, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        This function must perform iterative deepening if self.iterative=True,
        and it must use the search method (minimax or alphabeta) corresponding
        to the self.method value.

        **********************************************************************
        NOTE: If time_left < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        legal_moves : list<(int, int)>
            A list containing legal moves. Moves are encoded as tuples of pairs
            of ints defining the next (row, col) for the agent to occupy.

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """

        self.time_left = time_left

        # TODO: finish this function!

        # Perform any required initializations, including selecting an initial
        # move from the game board (i.e., an opening book), or returning
        # immediately if there are no legal moves
        NO_MOVE = (-1, -1)
        if len(legal_moves) == 0:
            return NO_MOVE
        loc = None
        try:
            # The search method call (alpha beta or minimax) should happen in
            # here in order to avoid timeout. The try/except block will
            # automatically catch the exception raised by the search method
            # when the timer gets close to expiring
            if self.method == 'minimax':
                fn = self.minimax
            elif self.method == 'alphabeta':
                fn = self.alphabeta
            else:
                raise Exception("Unknown Method")

            i = 1
            if self.iterative:
                while True:
                    _, loc = fn(game, i)
                    i += 1
            else:
                _, loc = fn(game, self.search_depth)

            return loc

        except Timeout:
            # Handle any actions required at timeout, if necessary
            return loc or NO_MOVE

    def minimax(self, game, depth, maximizing_player=True):
        """Implement the minimax search algorithm as described in the lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        #  finish this function!
        #raise NotImplementedError
        move_max=(-1,-1)
        legal_moves=game.get_legal_moves()
        if len(legal_moves)==0 or depth==0:
            #No more move
            return self.score(game,self),(-1,-1)
        if maximizing_player:
            score=-math.inf
            for move in legal_moves:
                new_game = game.forecast_move(move)
                if (depth==1) :
                     # Last level, get score
                    new_score=self.score(new_game,self)
                else:
                    # For this move test the next level
                    new_score,move_2= self.minimax(new_game,depth=depth-1,maximizing_player=False)
                if (new_score>score):
                    # Best score, update
                   score=new_score
                   move_max=move
        else:
            score=math.inf
            for move in legal_moves:
                new_game = game.forecast_move(move)
                if (depth==1) :
                    # Last level, get score
                    new_score=self.score(new_game,self)
                else:
                    # For this move test the next level
                    new_score,move_2=self.minimax(new_game,depth=depth-1)
                if (new_score<score):
                    # Best score, update
                   score=new_score
                   move_max=move

        return score, move_max


    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf"), maximizing_player=True):
        """Implement minimax search with alpha-beta pruning as described in the
        lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        # finish this function!
        #raise NotImplementedError
        move_max=(-1,-1)

        legal_moves=game.get_legal_moves()
        if len(legal_moves)==0 or depth==0:
            # No more move or last level
            return self.score(game,self),(-1,-1)
        if maximizing_player:
            v=-math.inf
            for move in legal_moves:
                new_game = game.forecast_move(move)
                if (depth==1):
                    #Last level, get value
                    new_score=self.score(new_game,self)
                else:
                    #For this movement test next level
                    new_score,move_2= self.alphabeta(new_game,depth=depth-1,maximizing_player=False,alpha=alpha,beta=beta)
                if (new_score>v):
                    v=new_score
                    move_max=move
                if v>=beta:
                    # Imposible to get a best solution,finish
                    return float(v),move_max
                alpha=max(alpha,v)
        else:
            v=math.inf
            for move in legal_moves:
                new_game = game.forecast_move(move)
                if (depth==1):
                    #Last level, get value
                    new_score=self.score(new_game,self)
                else:
                    #For this movement test next level
                    new_score,move_2=self.alphabeta(new_game,depth=depth-1,alpha=alpha,beta=beta)
                if (new_score<v):
                   v=new_score
                   move_max=move
                if v<=alpha:
                    # Imposible to get a best solution,finish
                    return float(v),move_max
                beta=min(beta,v)
        return float(v),move_max
