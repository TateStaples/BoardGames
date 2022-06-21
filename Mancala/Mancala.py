from pygame_resources import *
from copy import deepcopy
from time import time

board_len = 6
board_width = 1000
board_height = 500
home1 = 0
home2 = board_len + 1
avalanche = False
window = None

def setup_board():
    b = []
    for i in range(2):
        b.append(0)
        for i in range(board_len):
            b.append(2)
    return b

board = setup_board()

def get_locations():
    global board, board_height
    l = []
    l.append((board_width//(board_len+3), board_height//2))  # home1
    for i in range(home1+1, home2):  # side 1
        l.append((board_width//(board_len+3) * (i+1), board_height//4 * 3))  # side 1
    l.append((board_width-board_width//(board_len+3), board_height//2)) # home2
    for i in range(home2+1, len(board)):  # side 2
        l.append((board_width//(board_len+3) * (home2*2-i+1), board_height//4))  # side 2
    return l


def draw_board(board):
    global window
    size = 50
    if window is None:
        window = pygame.display.set_mode((board_width, board_height))
    window.fill(WHITE)
    for val, pos in zip(board, get_locations()):
        display(window, val, pos, size=size)
    pygame.display.update()


def advance(board, index, team, draw=False):
    first = True
    while board[index] > 0 and index != home1 and index != home2 and (avalanche or first):
        first = False
        val = board[index]
        board[index] = 0
        while val > 0:
            index = next(index, team)
            board[index] += 1
            val -= 1
            if draw:
                draw_board(board)
                display(window, val, (10, 10), 25, RED)
                pygame.display.update()
                freeze_display(0.5)

    new_team = not team if index != home1 and index != home2 else team
    return board, new_team


def valid_moves(board, team):
    moves = []
    if team == 0:
        for i in range(home1+1, home2):
            if board[i] > 0:
                moves.append(i)
    else:
        for i in range(home2+1, len(board)):
            if board[i] > 0:
                moves.append(i)
    return moves


def next(index, team): # team is 0 or 1
    if index == home2 - 1 and team != 0:
        return index+2
    if index == len(board)-1:
        return 1 if team else 0
    return index+1


def score(board):
    return board[home2] - board[home1]


def minimax_recursive(b, team, ply, alpha, beta):
    max_ply = 10
    moves = valid_moves(board, team)
    if ply >= max_ply or len(moves) == 0:
        return score(board), []
    if team == 0:
        current_max = -100000
        best_play = -1
        move_sequence = []
        for move in moves:
            modified = deepcopy(b)
            modified, new_team = advance(modified, move, team)
            value, moves = minimax_recursive(modified, new_team, ply + 1, alpha, beta)
            if value > current_max:
                current_max = value
                best_play = move
                moves.append(move)
                move_sequence = moves
                if alpha < current_max:
                    alpha = current_max
                if beta <= alpha:
                    break
        if ply == 0:
            return best_play, move_sequence
        return current_max, move_sequence
    else:  # MINI
        current_min = 100000
        best_play = -1
        move_sequence = []
        for move in moves:
            modified = deepcopy(b)
            modified, new_team = advance(modified, move, team)
            value, moves = minimax_recursive(modified, new_team, ply + 1, alpha, beta)
            if value < current_min:
                current_min = value
                best_play = move
                moves.append(move)
                move_sequence = moves
                if beta > current_min:
                    beta = current_min
                if beta <= alpha:
                    break
        if ply == 0:
            return best_play, move_sequence
        return current_min, move_sequence


def play_minimax():
    global board
    team = 0
    t = time()
    score, moves = minimax_recursive(board, team, 0, -100, 100)
    print(time()-t)
    print(moves)
    board = setup_board()
    for move in moves:
        board, team = advance(board, move, team, draw=True)
        freeze_display(0.5)


if __name__ == '__main__':
    play_minimax()
    freeze_display()

