"""
An AI player for Othello. 
"""

import random
import sys
import time

# You can use the functions in othello_shared to write your AI
from othello_shared import find_lines, get_possible_moves, get_score, play_move

cached_states = {}


def eprint(*args, **kwargs):  # you can use this for debugging, as it will print to sterr and not stdout
    print(*args, file=sys.stderr, **kwargs)


# Method to compute utility value of terminal state
def compute_utility(board, color):
    final_score = get_score(board)
    if color == 1:
        return final_score[0] - final_score[1]
    return final_score[1] - final_score[0]


# Better heuristic value of board
def compute_heuristic(board, color):
    return compute_heuristic_utility(board, color)


def weight_board(board):
    normal_weight = 1
    corner_weight = 5
    edge_weight = 3
    corner_neighbor_weight = -1
    center_weight = 1

    weight = []
    size = len(board)
    for i in range(size):
        weight.append([normal_weight] * size)

    # Setting the weight of the corner
    weight[0][0] = corner_weight
    weight[0][size - 1] = corner_weight
    weight[size - 1][0] = corner_weight
    weight[size - 1][size - 1] = corner_weight

    # Setting the weight of the edges, NOT corner OR neighbor
    for i in range(1, size - 1):
        weight[0][i] = edge_weight
        weight[size - 1][i] = edge_weight
        weight[i][0] = edge_weight
        weight[i][size - 1] = edge_weight

    # Setting the weight of the neighbor of the corner
    weight[0][1] = corner_neighbor_weight
    weight[1][0] = corner_neighbor_weight
    weight[0][size - 2] = corner_neighbor_weight
    weight[1][size - 1] = corner_neighbor_weight
    weight[size - 2][0] = corner_neighbor_weight
    weight[size - 1][1] = corner_neighbor_weight
    weight[size - 1][size - 2] = corner_neighbor_weight
    weight[size - 2][size - 1] = corner_neighbor_weight

    # Setting the weight of the center
    if size > 4:
        if size % 2 == 0:
            weight[size // 2][size // 2] = center_weight
            weight[size // 2][size // 2 - 1] = center_weight
            weight[size // 2 - 1][size // 2] = center_weight
            weight[size // 2 - 1][size // 2 - 1] = center_weight
        else:
            weight[size // 2][size // 2] = center_weight

    return weight


def get_heuristic_score(board, weight):
    color1_count = 0
    color2_count = 0
    for i in range(len(board)):
        for j in range(len(board)):
            if board[i][j] == 1:
                color1_count += weight[i][j]
            elif board[i][j] == 2:
                color2_count += weight[i][j]
    return color1_count, color2_count


def compute_heuristic_utility(board, color):
    weight = weight_board(board)
    color_1, color_2 = get_heuristic_score(board, weight)
    if color == 1:
        return color_1 - color_2
    return color_2 - color_1


def opponent(color):
    return 2 if color == 1 else 1


############ MINIMAX ###############################
def minimax_min_node(board, color, limit, caching=0):
    board_tuple = tuple(map(tuple, board))
    new_board = (color, board_tuple, limit)

    if caching == 1 and new_board in cached_states:
        return cached_states[new_board]

    if limit == 0 and caching == 1:
        cached_states[new_board] = (None, compute_utility(board, color))
        return None, compute_utility(board, color)

    possible_moves = get_possible_moves(board, opponent(color))
    if not possible_moves:
        output = (None, compute_utility(board, color))
        if caching == 1:
            cached_states[new_board] = output
        return output

    best_move = None
    min_utility = float('inf')
    for move in possible_moves:
        next_board = play_move(board, opponent(color), move[0], move[1])
        next_move, next_val = minimax_max_node(next_board, color, limit - 1, caching)
        if next_val < min_utility:
            min_utility = next_val
            best_move = move

    if caching == 1:
        cached_states[new_board] = best_move, min_utility
    return best_move, min_utility


def minimax_max_node(board, color, limit, caching=0):
    board_tuple = tuple(map(tuple, board))
    new_board = (color, board_tuple, limit)

    if caching == 1 and new_board in cached_states:
        return cached_states[new_board]

    if limit == 0:
        return None, compute_utility(board, color)

    possible_moves = get_possible_moves(board, color)
    if not possible_moves:
        output = (None, compute_utility(board, color))
        if caching == 1:
            cached_states[new_board] = output
        return output

    best_move = None
    max_utility = float('-inf')
    for move in possible_moves:
        next_board = play_move(board, color, move[0], move[1])
        next_move, next_val = minimax_min_node(next_board, color, limit - 1, caching)
        if next_val > max_utility:
            max_utility = next_val
            best_move = move

    if caching == 1:
        cached_states[new_board] = best_move, max_utility
    return best_move, max_utility


def select_move_minimax(board, color, limit, caching=0):
    """
    Given a board and a player color, decide on a move. 
    The return value is a tuple of integers (i,j), where
    i is the column and j is the row on the board.  

    Note that other parameters are accepted by this function:
    If limit is a positive integer, your code should enfoce a depth limit that is equal to the value of the parameter.
    Search only to nodes at a depth-limit equal to the limit.  If nodes at this level are non-terminal return a heuristic 
    value (see compute_utility)
    If caching is ON (i.e. 1), use state caching to reduce the number of state evaluations.
    If caching is OFF (i.e. 0), do NOT use state caching to reduce the number of state evaluations.    
    """
    move, _ = minimax_max_node(board, color, limit, caching)
    return move


############ ALPHA-BETA PRUNING #####################
def order_moves(moves, board, color, reverse=False):
    moves = sorted(moves, key=lambda move: compute_utility(play_move(board, color, move[0], move[1]), color),
                   reverse=reverse)
    return moves


def alphabeta_min_node(board, color, alpha, beta, limit, caching=0, ordering=0):
    board_tuple = tuple(map(tuple, board))
    new_board = (color, board_tuple, limit)

    if caching == 1 and new_board in cached_states:
        return cached_states[new_board]

    if limit == 0 and caching == 1:
        cached_states[new_board] = (None, compute_utility(board, color))
        return None, compute_utility(board, color)

    possible_moves = get_possible_moves(board, opponent(color))
    if not possible_moves:
        output = (None, compute_utility(board, color))
        if caching == 1:
            cached_states[new_board] = output
        return output

    best_move = None
    min_utility = float('inf')
    if ordering == 1:
        possible_moves = order_moves(possible_moves, board, color, reverse=False)
    for move in possible_moves:
        next_board = play_move(board, opponent(color), move[0], move[1])
        next_move, next_val = alphabeta_max_node(next_board, color, alpha, beta, limit - 1, caching, ordering)

        if next_val < min_utility:
            min_utility = next_val
            best_move = move
        if min_utility <= alpha:
            return best_move, min_utility
        beta = min(beta, min_utility)

    if caching == 1:
        cached_states[new_board] = best_move, min_utility
    return best_move, min_utility


def alphabeta_max_node(board, color, alpha, beta, limit, caching=0, ordering=0):
    board_tuple = tuple(map(tuple, board))
    new_board = (color, board_tuple, limit)

    if caching == 1 and new_board in cached_states:
        return cached_states[new_board]

    if limit == 0:
        return None, compute_utility(board, color)

    possible_moves = get_possible_moves(board, color)
    if not possible_moves:
        output = (None, compute_utility(board, color))
        if caching == 1:
            cached_states[new_board] = output
        return output

    best_move = None
    max_utility = float('-inf')
    if ordering == 1:
        possible_moves = order_moves(possible_moves, board, color, reverse=True)
    for move in possible_moves:
        next_board = play_move(board, color, move[0], move[1])
        next_move, next_val = alphabeta_min_node(next_board, color, alpha, beta, limit - 1, caching, ordering)
        if next_val > max_utility:
            max_utility = next_val
            best_move = move
        if max_utility >= beta:
            return best_move, max_utility
        alpha = max(alpha, max_utility)

    if caching == 1:
        cached_states[new_board] = (best_move, max_utility)
    return best_move, max_utility


def select_move_alphabeta(board, color, limit, caching=0, ordering=0):
    """
    Given a board and a player color, decide on a move. 
    The return value is a tuple of integers (i,j), where
    i is the column and j is the row on the board.  

    Note that other parameters are accepted by this function:
    If limit is a positive integer, your code should enfoce a depth limit that is equal to the value of the parameter.
    Search only to nodes at a depth-limit equal to the limit.  If nodes at this level are non-terminal return a heuristic 
    value (see compute_utility)
    If caching is ON (i.e. 1), use state caching to reduce the number of state evaluations.
    If caching is OFF (i.e. 0), do NOT use state caching to reduce the number of state evaluations.    
    If ordering is ON (i.e. 1), use node ordering to expedite pruning and reduce the number of state evaluations. 
    If ordering is OFF (i.e. 0), do NOT use node ordering to expedite pruning and reduce the number of state evaluations. 
    """
    alpha = float('-inf')
    beta = float('inf')

    move, _ = alphabeta_max_node(board, color, alpha, beta, limit, caching, ordering)
    return move


####################################################
def run_ai():
    """
    This function establishes communication with the game manager.
    It first introduces itself and receives its color.
    Then it repeatedly receives the current score and current board state
    until the game is over.
    """
    print("Othello AI")  # First line is the name of this AI
    arguments = input().split(",")

    color = int(arguments[0])  # Player color: 1 for dark (goes first), 2 for light.
    limit = int(arguments[1])  # Depth limit
    minimax = int(arguments[2])  # Minimax or alpha beta
    caching = int(arguments[3])  # Caching
    ordering = int(arguments[4])  # Node-ordering (for alpha-beta only)

    if (minimax == 1):
        eprint("Running MINIMAX")
    else:
        eprint("Running ALPHA-BETA")

    if (caching == 1):
        eprint("State Caching is ON")
    else:
        eprint("State Caching is OFF")

    if (ordering == 1):
        eprint("Node Ordering is ON")
    else:
        eprint("Node Ordering is OFF")

    if (limit == -1):
        eprint("Depth Limit is OFF")
    else:
        eprint("Depth Limit is ", limit)

    if (minimax == 1 and ordering == 1): eprint("Node Ordering should have no impact on Minimax")

    while True:  # This is the main loop
        # Read in the current game status, for example:
        # "SCORE 2 2" or "FINAL 33 31" if the game is over.
        # The first number is the score for player 1 (dark), the second for player 2 (light)
        next_input = input()
        status, dark_score_s, light_score_s = next_input.strip().split()
        dark_score = int(dark_score_s)
        light_score = int(light_score_s)

        if status == "FINAL":  # Game is over.
            print
        else:
            board = eval(input())  # Read in the input and turn it into a Python
            # object. The format is a list of rows. The
            # squares in each row are represented by
            # 0 : empty square
            # 1 : dark disk (player 1)
            # 2 : light disk (player 2)

            # Select the move and send it to the manager
            if (minimax == 1):  # run this if the minimax flag is given
                movei, movej = select_move_minimax(board, color, limit, caching)
            else:  # else run alphabeta
                movei, movej = select_move_alphabeta(board, color, limit, caching, ordering)

            print("{} {}".format(movei, movej))


if __name__ == "__main__":
    run_ai()
