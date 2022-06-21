# october / november 2019

import pygame

pygame.init()

windowWidth = 1200
windowHeight = 700
window = pygame.display.set_mode((windowWidth, windowHeight))

RED = (255, 0, 0)
YELLOW = (255, 255, 0)
BLUE = (0, 0, 255)
WHITE = (255, 255, 255)

board = [
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0]
]


def stall(how_long):
    clock = pygame.time.Clock()
    time = 0
    while True:
        time += clock.tick()
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                quit()
        if time >= how_long:
            return
        pygame.display.update()


def list_in(sub, super):
    for start_index in range(len(super)-len(sub) + 1):
        slice = []
        for count in range(len(sub)):
            slice.append(super[start_index+count])
        if slice == sub:
            return True
    return False


def draw_board():
    window.fill(WHITE)
    pygame.draw.rect(window, BLUE, [100, 100, 1000, 550])
    for row_count, row in enumerate(board):
        for col, spot in enumerate(row):
            if spot == 0: color = WHITE
            elif spot == 1: color = RED
            else: color = YELLOW
            pygame.draw.circle(window, color, (225 + 125 * col, 150 + row_count * 80), 30)
    pygame.display.update()


def move(a_board, col, player):
    if a_board[0][col] != 0:
        print("invalid move")
        return a_board
    for count, row in enumerate(a_board[::-1]):
        if row[col] == 0:
            a_board[len(a_board) - count - 1][col] = player
            break
    return a_board


def board_state(board, player):
    if player == 1:
        not_player = 2
    else:
        not_player = 1

    # check horizontal wins
    for row in board:
        if list_in([player, player, player, player], row):
            return 1
        elif list_in([not_player, not_player, not_player, not_player], row):
            return -1

    # check for vertical wins
    for col_count in range(7):
        col = [row[col_count] for row in board]
        if list_in([player, player, player, player], col):
            return 1
        elif list_in([not_player, not_player, not_player, not_player], col):
            return -1

    # check pos diagonals
    for row_index in range(3, 6):
        for col_index in range(4):
            diagonal = [board[row_index-i][col_index+i] for i in range(4)]
            if list_in([player, player, player, player], diagonal):
                return 1
            elif list_in([not_player, not_player, not_player, not_player], diagonal):
                return -1

    # check neg diagonals
    for row_index in range(3):
        for col_index in range(4):
            diagonal = [board[row_index+i][col_index+i] for i in range(4)]
            if list_in([player, player, player, player], diagonal):
                return 1
            elif list_in([not_player, not_player, not_player, not_player], diagonal):
                return -1
    return 0


def game_over(board):
    if board_state(board, 1) != 0:
        return True
    for spot in board[0]:
        if spot == 0:
            return False
    return True


def copy(board):
    new_board = []
    for row in board:
        new_row = []
        for col in row:
            new_row.append(col)
        new_board.append(new_row)
    return new_board


def evaluate_board(board, player):
    state = board_state(board, player)
    if state != 0:
        return state * 500
    score = 0
    board_values = [
        [3, 4, 5, 7, 5, 4, 3],
        [4, 6, 8, 10, 8, 6, 4],
        [5, 8, 11, 13, 11, 8, 5],
        [5, 8, 11, 13, 11, 8, 5],
        [4, 6, 8, 10, 8, 6, 4],
        [3, 4, 5, 7, 5, 4, 3]
    ]

    for row in range(len(board)):
        for col in range(len(board[0])):
            if board[row][col] == player:
                score += board_values[row][col]
            elif board[row][col] != 0:
                score -= board_values[row][col]
    return score


def Mini_Max(the_board, player, current_ply, max_ply, alpha, beta, og):
    if current_ply == max_ply or game_over(the_board):
        return evaluate_board(the_board, og)
    other = [2, 1][player-1]
    if current_ply % 2 == 0:  # MAX
        current_max = -100000
        best_play = -1
        for col in range(7):
            if the_board[0][col] == 0:
                modified = copy(the_board)
                modified = move(modified, col, player)
                value = Mini_Max(modified, other, current_ply + 1, max_ply, alpha, beta, og)
                if value > current_max:
                    current_max = value
                    best_play = col
                    if alpha < current_max:
                        alpha = current_max
                    if beta <= alpha:
                        break
        if current_ply == 0:
            global board
            board = move(board, best_play, og)
        return current_max
    else:  # MINI
        current_min = 100000
        for col in range(7):
            if the_board[0][col] == 0:
                modified = copy(the_board)
                modified = move(modified, col, player)
                value = Mini_Max(modified, other, current_ply + 1, max_ply, alpha, beta, og)
                if value < current_min:
                    current_min = value
                    if beta > current_min:
                        beta = current_min
                    if beta <= alpha:
                        break

        return current_min


def main():
    global board
    player = 1
    while not game_over(board):
        if player == 1: color = RED
        else: color = YELLOW
        clicked = None
        while clicked is None:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    quit()

                if event.type == pygame.MOUSEBUTTONDOWN:
                    clicked = (pygame.mouse.get_pos()[0] - 100) // 140
                    if clicked < 0 or clicked > 6:
                        clicked = None

            mouse_location = pygame.mouse.get_pos()[0]
            draw_board()
            pygame.draw.circle(window, color, (mouse_location, 50), 30)
            pygame.display.update()
        board = move(board, clicked, player)
        status = board_state(board, player)
        #Mini_Max(board, 1, 0, 5, -100, 100, 1)
        draw_board()
        stall(1)
        if game_over(board):
            break
        # go to next player
        Mini_Max(board, 2, 0, 5, -100, 100, 2)
        draw_board()
        stall(1000)
    answers = ["draw", "you win", "you lose"]
    print(answers[status])


if __name__ == '__main__':
    main()

    # this is so the app doesn't quit when over
    pygame.display.update()
    while True:
        pygame.time.delay(1000)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit()
