"""Finish all TODO items in this file to complete the isolation project, then
test your agent's strength against a set of known agents using tournament.py
and include the results in your report.
"""
import random
from isolation import Board



class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass


def custom_score(game, player):
    """Attempts to maximize the total number of moves made in the game. This selects
    the longest path that the agent can take unless an earlier move creates a win.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : hashable
        One of the objects registered by the game object as a valid player.
        (i.e., `player` should be either game.__player_1__ or
        game.__player_2__).

    Returns
    ----------
    float
        The heuristic value of the current game state
 """   
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")
    
    return float(game.move_count)
        
     
def custom_score_2(game, player):
    """Returns a positive value if the forecasted move is in the opponent's legal
    move list.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : hashable
        One of the objects registered by the game object as a valid player.
        (i.e., `player` should be either game.__player_1__ or
        game.__player_2__).

    Returns
    ----------
    float
        The heuristic value of the current game state
    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")
 
    if game.get_player_location(player) in game.get_legal_moves(game.get_opponent(player)):
        return 100.0
    
    return 10.0
    

def custom_score_3(game, player):
    """Returns the negative value of the distance away from the opponent. This
    causes the agent to try to stay near the opponent.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : hashable
        One of the objects registered by the game object as a valid player.
        (i.e., `player` should be either game.__player_1__ or
        game.__player_2__).

    Returns
    ----------
    float
        The heuristic value of the current game state
    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")
    
    w, h = game.get_player_location(game.get_opponent(player))
    y, x = game.get_player_location(player)
    return float(-1*((h - y)**2 + (w - x)**2))

def custom_score_4(game, player):
    """Custom score 1 + the improved_score function with a weighted opponent
    move count.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : hashable
        One of the objects registered by the game object as a valid player.
        (i.e., `player` should be either game.__player_1__ or
        game.__player_2__).

    Returns
    ----------
    float
        The heuristic value of the current game state
 """   
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")
    
    return float(game.move_count + len(game.get_legal_moves(player)) - 2.0 * len(game.get_legal_moves(game.get_opponent(player))) )
    

class IsolationPlayer:
    """Base class for minimax and alphabeta agents -- this class is never
    constructed or tested directly."""
    
    def __init__(self, search_depth=3, score_fn=custom_score, timeout=10.):
        self.search_depth = search_depth
        self.score = score_fn
        self.time_left = None
        self.TIMER_THRESHOLD = timeout


class MinimaxPlayer(IsolationPlayer):

    def get_move(self, game, time_left):
                
        self.time_left = time_left
        best_move = game.get_legal_moves()

        try:
            return self.minimax(game, self.search_depth)

        except SearchTimeout:
            return best_move[0]  

        return best_move[0]

    def minimax(self, game, depth):
      
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        player = game._active_player
        
        def max_value(game, depth):
            if self.time_left() < self.TIMER_THRESHOLD:
                raise SearchTimeout()
            
            if depth == 0 or game.is_winner(player) or game.is_loser(player):
                return self.score(game, self)
            v = float("-inf")
            for a in game.get_legal_moves():
                v = max(v, min_value(game.forecast_move(a), depth - 1 ))            
            return v
                   
        def min_value(game, depth):
            if self.time_left() < self.TIMER_THRESHOLD:
                raise SearchTimeout()
            
            if depth == 0 or game.is_winner(player) or game.is_loser(player):
                return self.score(game, self)
            v = float("inf")
            for a in game.get_legal_moves():
                v = min(v, max_value(game.forecast_move(a), depth - 1))
            return v
        
        
        legal_moves = game.get_legal_moves()
        if legal_moves:
            best_move = legal_moves[0]
            best_score = float("-inf")
            for a in legal_moves:
                v = min_value(game.forecast_move(a), depth - 1)
                if v > best_score:
                    best_score = v
                    best_move = a
            return best_move
        
class AlphaBetaPlayer(IsolationPlayer):

    def get_move(self, game, time_left):
               
        self.time_left = time_left
        depth = self.search_depth
        best_move = game.get_legal_moves()
        
        while True:
            try:
                best_move = self.alphabeta(game, depth)
                depth += 1
            except SearchTimeout:
                return best_move
        
        return best_move[0]
        

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf")):
    
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        
        player = game._active_player
        
        def max_value(game, alpha, beta, depth):
            if self.time_left() < self.TIMER_THRESHOLD:
                raise SearchTimeout()
            
            if depth == 0 or game.is_winner(player) or game.is_loser(player):
                return self.score(game, self)
            v = float("-inf")
            for a in game.get_legal_moves():
                v = max(v, min_value(game.forecast_move(a), alpha, beta, depth - 1 ))
                if v >= beta:
                    return v
                alpha = max(alpha, v)
            return v
                   
        def min_value(game, alpha, beta, depth):
            if self.time_left() < self.TIMER_THRESHOLD:
                raise SearchTimeout()
            
            if depth == 0 or game.is_winner(player) or game.is_loser(player):
                return self.score(game, self)
            v = float("inf")
            for a in game.get_legal_moves():
                v = min(v, max_value(game.forecast_move(a), alpha, beta, depth - 1))
                if v <= alpha:
                    return v
                beta = min(beta, v)
            return v
        
        legal_moves = game.get_legal_moves()
        if legal_moves:
            best_move = legal_moves[0]
            best_score = float("-inf")
            beta = float("inf")
            for a in legal_moves:
                v = min_value(game.forecast_move(a), best_score, beta, depth - 1)
                if v > best_score:
                    best_score = v
                    best_move = a
            return best_move
        
