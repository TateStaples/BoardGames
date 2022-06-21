from Scrabble.image_processing import Image_processor
from copy import deepcopy
'''
Limitations:
- photo can't deal with blank
- can't do multipliers
- photo not get all
'''


class ScrabblePlayer:
    cols = 15
    rows = 15

    scores = {
        'a': 1,
        'b': 3,
        'c': 3,
        'd': 2,
        'e': 1,
        'f': 4,
        'g': 2,
        'h': 4,
        'i': 1,
        'j': 8,
        'k': 5,
        'l': 1,
        'm': 3,
        'n': 1,
        'o': 1,
        'p': 3,
        'q': 10,
        'r': 1,
        's': 1,
        't': 1,
        'u': 1,
        'v': 4,
        'w': 4,
        'x': 8,
        'y': 4,
        'z': 10,
    }
    path_to_words = "words.txt"

    def __init__(self, img_path, hand, testing=False):
        self.board = [[" " for i in range(self.cols)] for i in range(self.rows)]
        self.words = []
        self.start_indices = {}  # index of the start of each letter
        decoder = Image_processor(img_path)
        self.board = decoder.run()
        self.hand = hand
        self.best_score = 0
        self.best_play = None
        self.nn = None
        self.draw()
        if not testing:
            self.activate()

    def activate(self):
        self.load_dictionary()
        print("words loaded")
        # self.create_neural_network()
        # print("network trained")
        # self.template_process()
        print("image loaded")
        self.find_best_move()
        print("letter found")
        if self.best_play is not None:
            self.place(self.best_play)
        self.draw()

    def load_dictionary(self):
        previous_letter = ""
        with open(self.path_to_words, 'r') as file:
            for i, line in enumerate(file):
                line = line.strip()
                self.words.append(line)
                if line[0] != previous_letter:
                    previous_letter = line[0]
                    self.start_indices[line[0]] = i

    def find_best_move(self):
        moves = self.get_possible_moves()
        print(len(moves))
        for r, c in moves:
            print("tick")
            # vertical
            playable_area = self.ray_cast(r, c, "up")[::-1] + self.ray_cast(r, c, "up")[1:]
            options = self.possible_words(playable_area)
            for word in options:
                for play in self.word_to_play(word, playable_area, (r, c), "vertical"):
                    if play is None:
                        continue
                    score = self.score_play(play)
                    if score > self.best_score:
                        self.best_score = score
                        self.best_play = play

            # horizontal
            playable_area = self.ray_cast(r, c, "left")[::-1] + self.ray_cast(r, c, "right")[1:]
            options = self.possible_words(playable_area)
            for word in options:
                for play in self.word_to_play(word, playable_area, (r, c), "horizontal"):
                    score = self.score_play(play)
                    if score > self.best_score:
                        self.best_score = score
                        self.best_play = play

    def get_possible_moves(self):
        spots = []
        for row in range(len(self.board)):
            for col in range(len(self.board[row])):
                letter = self.board[row][col]
                if letter != " ":
                    neighbors = self.get_neighbors(row, col)
                    for r, c in neighbors:
                        new = self.board[r][c]
                        if new == " ":
                            spots.append((r, c))
        return spots

    def possible_words(self, playable):
        possible = []
        for word in self.words:
            if self.is_word(word) and self.fits(word, playable) is not None:
                possible.append(word)
        return possible

    def ray_cast(self, r, c, direction, dis=7):
        string = ""
        amount_of_spaces = 0
        if direction == "up":
            while amount_of_spaces <= dis:
                r -= 1
                if not 0 < r < len(self.board):
                    return string
                letter = self.board[r][c]
                if amount_of_spaces == 0:
                    string += "_"
                    amount_of_spaces += 1
                elif letter == " ":
                    string += letter
                    amount_of_spaces += 1
                else:
                    string += self.get_word(r, c, direction)
            return string
        elif direction == "down":
            while amount_of_spaces <= dis:
                r += 1
                if not 0 < r < len(self.board):
                    return string
                letter = self.board[r][c]
                if letter == " ":
                    string += letter
                    amount_of_spaces += 1
                else:
                    string += self.get_word(r, c, direction)
            return string
        elif direction == "left":
            while amount_of_spaces <= dis:
                c -= 1
                if not 0 < c < len(self.board):
                    return string
                letter = self.board[r][c]
                if letter == " ":
                    string += letter
                    amount_of_spaces += 1
                else:
                    string += self.get_word(r, c, direction)
            return string
        else:  # right
            while amount_of_spaces <= dis:
                c += 1
                if not 0 < c < len(self.board[r]):
                    return string
                letter = self.board[r][c]
                if letter == " ":
                    string += letter
                    amount_of_spaces += 1
                else:
                    string += self.get_word(r, c, direction)
            return string

    def get_word(self, r, c, direction):
        if direction == "up":
            word = ""
            letter = self.board[r][c]
            while letter != " ":
                word += letter
                r -= 1
                if r < 0:
                    return word
                letter = self.board[r][c]
            return word
        elif direction == "down":
            word = ""
            letter = self.board[r][c]
            while letter != " ":
                word += letter
                r += 1
                if r >= len(self.board):
                    return word
                letter = self.board[r][c]
            return word
        elif direction == "left":
            word = ""
            letter = self.board[r][c]
            while letter != " ":
                word += letter
                c -= 1
                if c < 0:
                    return word
                letter = self.board[r][c]
            return word
        else:  # right
            word = ""
            letter = self.board[r][c]
            while letter != " ":
                word += letter
                c += 1
                if c >= len(self.board[r]):
                    return word
                letter = self.board[r][c]
            return word

    def is_word(self, word):  # cant binary in python
        if len(word) == 1:
            return True
        for wrd in self.words:
            if word == wrd:
                return True
        return False

    def can_make(self, word, letters):
        blanks = 0
        while " " in letters:  # strip blanks
            blanks += 1
            letters.remove(" ")
        hist1 = self.letter_hist(word)
        hist2 = self.letter_hist(letters)
        for c1, c2 in zip(hist1, hist2):
            if c2 < c1:  # if not enough letter
                blanks -= c1 - c2
                if blanks < 0:
                    return False
        return True

    def word_to_play(self, word, playable_area, origin, direction):
        r, c = origin
        start_index = self.fits(word, playable_area)
        if start_index is None:
            return ()
        slice = playable_area[start_index: start_index + len(word)]
        origin_index = slice.index("_")
        play = []
        if direction == "vertical":
            r -= origin_index
            for letter in word:
                play.append(((r, c), letter))
                r += 1
        else:
            c -= origin_index
            for letter in word:
                play.append(((r, c), letter))
                c += 1
        return play

    def get_neighbors(self, r, c):
        neighbors = []
        if r > 0:
            neighbors.append((r-1, c))
        if r < len(self.board)-1:
            neighbors.append((r+1, c))
        if c > 0:
            neighbors.append((r, c-1))
        if c < len(self.board)-1:
            neighbors.append((r, c+1))
        return neighbors

    def score_play(self, play):
        og = deepcopy(self.board)
        og_h = ""
        og_v = ""
        self.place(play)
        score = 0
        for spot, letter in play:
            r, c = spot
            left = self.get_word(r, c, "left")[::-1]
            right = self.get_word(r, c, "right")
            up = self.get_word(r, c, "up")[::-1]
            down = self.get_word(r, c, "down")
            horizontal = left + right[1:] if len(right) > 1 else left
            vertical = up + down[1:] if len(down) > 1 else up
            if not (self.is_word(horizontal) or self.is_word(vertical)):
                self.board = og
                return -1  # invalid play
            if score == 0:
                og_h = horizontal
                og_v = vertical
            else:
                if horizontal == og_h:
                    horizontal = ""
                if vertical == og_v:
                    vertical = ""
            score += self.score_word(horizontal) if len(horizontal) > 1 else 0
            score += self.score_word(vertical) if len(vertical) > 1 else 0
        self.board = og
        return score

    def place(self, play):
        for spot, letter in play:
            r, c = spot
            # print(spot)
            self.board[r][c] = letter

    def score_word(self, word):
        return sum([self.scores[l.lower()] for l in word])

    @staticmethod
    def letter_hist(word):
        blank = [0 for i in range(26)]
        for letter in word:
            index = ord(letter) - 97
            blank[index] += 1
        return blank

    def fits(self, word, playable_area):
        length = len(word)
        for i in range(len(playable_area)-length+1):
            through_center = False
            letters = deepcopy(self.hand)
            slice = playable_area[i:i+length]
            if "_" not in slice:
                continue
            for letter, fill_spot in zip(word, slice):
                if fill_spot == " ":
                    if letter in letters:
                        letters.remove(letter)
                    else:
                        break
                elif fill_spot == "_":
                    through_center = True
                    if letter in letters:
                        letters.remove(letter)
                    else:
                        break
                elif letter != fill_spot:
                    break
            else:
                if through_center:
                    return i
        return None

    def draw(self):
        import pygame
        pygame.init()
        windowWidth = 500
        windowHeight = 500
        window = pygame.display.set_mode((windowWidth, windowHeight))
        row_size = windowHeight / len(self.board)
        col_size = windowWidth // len(self.board[0])
        new_spots = [spot for spot, l in self.best_play] if self.best_play is not None else []
        for r in range(self.rows):
            for c in range(self.cols):
                x, y = c*col_size, r*row_size
                color = (255, 0, 0) if (r, c) in new_spots else (0, 0, 0)
                pygame.draw.rect(window, (255, 255, 255), (x, y, col_size-1, row_size-1))
                font = pygame.font.SysFont(pygame.font.get_fonts()[0], 20)
                score_surface = font.render(self.board[r][c], False, color)
                window.blit(score_surface, (x, y))
        pygame.display.update()
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()

    def play_2(self):
        best_play = None
        best_score = -1
        for word in self.words:
            play = self.score_2(word)
            s = play.score
            if s > best_score:
                best_play = play
                best_score = s

    def score_2(self, word):
        increments = ((1, 0), (0, 1))
        # allow not start at predefined
        alphabet = 'abcdefghijklmnopqrstuvwxyz'
        play = None
        for r in range(self.rows):
            for c in range(self.cols):
                if self.board[r][c] in word:
                    for dr, dc in increments:  # horizontl and veritcal
                        hand = self.letter_hist(self.hand)
                        works = True
                        i = word.index(self.board[r][c])
                        new_r, new_c = r, c
                        while i < len(word):
                            new_r += dr
                            new_c += dc
                            if self.board[new_r][new_c] == word[i]:
                                i += 1
                            elif hand[alphabet.index(self.board[new_r][new_c])] > 0:
                                hand[alphabet.index(self.board[new_r][new_c])] -= 1
                            else:
                                works = False
                                break
                            i += 1
                        if works:
                            score = self.score_word(word)
                            if play is None or play.score < score:
                                play = Play(word, (r, c), "h" if dr == 0 else "v", score)
        return play


class Play:
    def __init__(self, word, start, direction, score):
        self.word = word
        self.start = start
        self.direction = "h" if direction[0] == "h" else "v"
        self.score = score

if __name__ == '__main__':
    main = ScrabblePlayer("data/real_board.jpg", ["a", "b", "c"])


