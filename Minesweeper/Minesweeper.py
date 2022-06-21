import pygame
import random

# This is a minesweeper game that can be solved manually or by an AI
# December 4, 2019
__author__ = "Tate Staples"

# todo: do thing with adding sums of neighbors
# todo fix werid fill thing

windowWidth = 800
windowHeight = 800
window = pygame.display.set_mode((windowWidth, windowHeight))

row_count = 30
col_count = 30
bomb_count = 80

# A bunch  of color rgb values
WHITE = (255, 255, 255)
YELLOW = (255, 255, 0)
BLUE = (0, 0, 255)
RED = (255, 0, 0)
PINK = (255, 192, 203)
ORANGE = (255, 165, 0)
LIGHT_BLUE = (135, 206, 235)
GREEN = (0, 255, 0)
BROWN = (165, 42, 42)
BLACK = (0, 0, 0)

bomb_cords = [(random.randint(0, row_count-1), random.randint(0, col_count-1)) for i in range(bomb_count)]
run = True

# establish value of each square
board_vals = []
flags = []
for r in range(row_count):
    row_values = []
    row_flags = []
    for c in range(col_count):
        row_flags.append(False)
        if (r, c) in bomb_cords:
            row_values.append("bomb")
        else:
            surrounding_bombs = 0
            for i in range(-1, 2):
                for j in range(-1, 2):
                    if (r+i, c+j) in bomb_cords:
                        surrounding_bombs += 1
            row_values.append(surrounding_bombs)
    board_vals.append(row_values)
    flags.append(row_flags)

# create starter display
for r in range(row_count+1):
    for c in range(col_count+1):
        row_height = windowHeight // row_count
        col_width = windowWidth // col_count
        x_cord = col_width * c
        y_cord = row_height * r
        if (r, c) in bomb_cords and False:  # this is used for debugging
            pygame.draw.rect(window, PINK, (x_cord, y_cord, col_width - 1, row_height - 1))
        else:
            pygame.draw.rect(window, LIGHT_BLUE, (x_cord, y_cord, col_width-1, row_height-1))
pygame.display.update()


# converts clicked location to row/col
def get_square(cords):
    global windowWidth, windowHeight, row_count, col_count
    x, y = cords
    row_height = windowHeight // row_count
    col_width = windowWidth // col_count
    row = y // row_height
    col = x // col_width
    return row, col


# flags a spot that player thinks is a bomb
def flag(spot):
    global flags
    row, col = spot
    row_height = windowHeight // row_count
    col_width = windowWidth // col_count
    x_cord = col_width * col
    y_cord = row_height * row
    if flags[row][col]:
        pygame.draw.rect(window, LIGHT_BLUE, (x_cord, y_cord, col_width - 1, row_height - 1))
        flags[row][col] = False
    else:
        pygame.draw.rect(window, RED, (x_cord, y_cord, col_width-1, row_height-1))
        flags[row][col] = True


# reveals a clicked square
revealed = []  # list of revealed coordinates
def reveal(cords):
    global board_vals, revealed, windowWidth, windowHeight, row_count, col_count, run
    row, col = cords
    val = board_vals[row][col]
    if val == "bomb":
        print("You hit a bomb, LOOSER!")
        print(cords)
        for row in calculate_probs():
            for val in row:
                print(round(val, 1), end="\t")
            print()
        run = False
        return
        #quit()
    revealed.append((row, col))

    row_height = windowHeight // row_count
    col_width = windowWidth // col_count
    x_cord = col_width * col
    y_cord = row_height * row

    text_font = 'Comic Sans MS'
    # points = pixels * 72 / 96
    pixels = row_height if row_height < col_width else col_width
    text_size = int(pixels * 72 / 48)
    font = pygame.font.SysFont(text_font, text_size)
    score_surface = font.render(str(val), False, RED)

    window.blit(score_surface, (x_cord, y_cord))

    if val == 0:  # this section reveals large area
        for neighbor in get_neighbors(cords):
            if neighbor not in revealed and neighbor not in locked:
                try:
                    reveal(neighbor)
                except Exception as e:
                    print(f"Exeception: {e}")


# returns a list of all surround boxes
def get_neighbors(cords):
    neighbors = []
    row, col = cords
    for i in range(-1, 2):
        for j in range(-1, 2):
            if i == 0 and j == 0:
                continue
            new_row = row + i
            new_col = col + j
            if new_col < 0 or new_col >= len(board_vals[0]):  # if not in a valid column
                continue
            if new_row < 0 or new_row >= len(board_vals):  # if not in a valid row
                break
            neighbors.append((new_row, new_col))
    return neighbors


locked = [] # list of spots that aren't worth checking anymore
known_bombs = []  # list of previously identified bombs
def calculate_probs():
    global revealed, board_vals, known_bombs, locked
    # step 1: create random
    bomb_probs = []
    try:
        default_value = bomb_count / (row_count * col_count - len(revealed) - len(known_bombs))
    except ZeroDivisionError:  # this occurs when you win
        print("over")
        return
    for row in range(row_count):  # creates initial probabilities
        row_probs = []
        for col in range(col_count):
            row_probs.append(default_value)
        bomb_probs.append(row_probs)

    # step 1: create locked
    for r, c in locked:
        bomb_probs[r][c] = 2.0

    # step 2: establish already found bombs
    for r, c in known_bombs:
        bomb_probs[r][c] = 1.0

    # step 3: establish simple probs
    for row, col in revealed:
        val = board_vals[row][col]
        bomb_probs[row][col] = 2.0  # can't be a already revealed number
        bomb_probs = update_probs((row, col), val, bomb_probs, default_value)

    # step 4: use known bombs
    is_bomb = True
    while is_bomb:
        is_bomb = False
        for row, r in enumerate(bomb_probs):
            for col, c in enumerate(r):
                if c == 1.0 and (row, col) not in known_bombs:
                    is_bomb = True
                    known_bombs.append((row, col))
                    flag((row, col))  # this is weird late game
                    neighbors = get_neighbors((row, col))
                    for neighbor in neighbors:
                        if neighbor in revealed:
                            val = board_vals[neighbor[0]][neighbor[1]]
                            bomb_probs = update_probs(neighbor, val, bomb_probs, default_value)
    return bomb_probs


# calculates probablities
def update_probs(cords, val, bomb_probs, default_val):
    global known_bombs, locked
    neighbors = get_neighbors(cords)
    amount_of_known = 0  # amount of revealed squares
    amount_of_unknown = 0  # amount of uncertain squares
    amount_of_bombs = 0  # amount of already found bombs
    blanks = []
    for neighbor in neighbors:  # get a count of each type
        r, c = neighbor
        if neighbor in revealed or bomb_probs[r][c] == 0 or neighbor in locked:  # if the neighbor is a known
            amount_of_known += 1
        elif bomb_probs[r][c] == 1.0 or (r, c) in known_bombs:  # if there is a bomb there, 100% chance
            amount_of_bombs += 1
        else:
            amount_of_unknown += 1
            blanks.append(neighbor)
    try:
        prob = (val - amount_of_bombs) / amount_of_unknown  # formula for probability
    except ZeroDivisionError:  # occurs when nothing unknown around it
        locked.append(cords)  # this saves this coordinate so it doesn't have to be recalculated every time
        revealed.remove(cords)
        return bomb_probs
    for spot in blanks:
        r, c = spot
        if bomb_probs[r][c] < prob or bomb_probs[r][c] == default_val or prob == 0:  # checks if it should override current prob
            bomb_probs[r][c] = prob
    return bomb_probs


# AI to solve
def solve():
    global revealed, bomb_count, board_vals, row_count, col_count, run

    while run:  # main loop
        bomb_probs = calculate_probs()  # solves for all probabilities
        if bomb_probs is None:  # prevents a bug that happens occasionally
            break
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit()
        pygame.time.delay(0) # this can be raised if you want the AI to slow down so people can watch

        # finds the lowest value
        spot = None  # initializes value
        minimum = 1
        for r, row in enumerate(bomb_probs):
            low = min(row)
            i = row.index(low)
            if low < minimum:
                minimum = low
                spot = r, i

        if spot is not None:  # prevents possible exception
            reveal(spot)  # clicks the lowest spot
            pygame.display.update()  # updates the display


def user_input():
    global run
    while run:
        has_clicked = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            if event.type == pygame.MOUSEBUTTONDOWN:
                clicked = get_square(pygame.mouse.get_pos())
                reveal(clicked)
                has_clicked = True
        keys = pygame.key.get_pressed()
        if keys[pygame.K_SPACE]:
            location = pygame.mouse.get_pos()
            square = get_square(location)
            flag(square)
            has_clicked = True
        if keys[pygame.K_RETURN]:
            solve()
            run = False
        if has_clicked:
            pygame.display.update()


# main method that takes input and redirects to solve type
def main():
    choice = input("AI or player? ")  # user input
    pygame.init()

    if 'ai' in choice.lower():
        solve()
    else:
        user_input()


if __name__ == "__main__":
    main()

    # this is so the app doesn't quit when over
    pygame.display.update()
    while True:
        pygame.time.delay(1000)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit()
            if event.type == pygame.KEYDOWN:
                print("Test")
                main()
