from random import choice
from time import sleep
from copy import deepcopy
pygame = None
pygame_imported = False
window = None
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLUE = (0, 255, 0)
GREEN = (0, 0, 255)


def import_pygame():
    global pygame_imported, window, pygame
    pygame_imported = True
    import pygame as pg
    pg.init()
    window = pg.display.set_mode((500, 500))
    pg.display.set_caption('Chess game')
    pygame = pg


class ChessPosition:
    board_size = 8
    general_id = 0

    def __init__(self, b=None):
        self.board = b if b is not None else self.get_default_board()
        self.active_team = WHITE
        self.id = self.general_id
        self.general_id += 1

    def play_random_game(self, depth=10, draw=True):
        move = None
        for i in range(depth):
            m = self.make_random_move()
            if i == 0:
                move = m
            if draw:
                self.display()
                self.draw_valid_moves()
                sleep(1)
        return move

    def play_montecarlo(self, draw=True):
        while not self.gameover():
            move = self.montecarlo(500)
            self.move(move)
            if draw:
                print("drawing")
                self.display()
                # self.draw_valid_moves()
                # sleep(1)
        print("game over")
        sleep(10)

    def play_minimax(self, draw=False):
        while True:
            val, move = self.minimax(5)
            self.move(move)
            if draw:
                self.display()

    def gameover(self):
        king_count = 0
        for row in self.board:
            for square in row:
                if square is not None and type(square) is King:
                    king_count += 1
        return king_count < 2

    def score(self):
        num = 0
        for row in self.board:
            for square in row:
                if square is not None:
                    num += square.val if square.team == WHITE else -square.val
        return num

    def montecarlo(self, quantity):
        scores = {}
        moves = {}
        for game in range(quantity):
            new_board = self.copy()
            m = new_board.play_random_game(depth=5, draw=False)
            s = new_board.score()
            string = str(m)
            s *= 1 if self.active_team == WHITE else -1
            if string in scores:
                scores[string] += s
            else:
                scores[string] = s
                moves[string] = m
            del new_board
        top_score = max(scores.values())
        for key in scores:
            if scores[key] == top_score:
                return moves[key]

    def minimax(self, ply_left, alpha=-1000, beta=1000):  # copyed from Connect Four, need to check
        if ply_left <= 0 or self.gameover():
            return self.score(), None
        if self.active_team == WHITE:  # MAX
            current_max = -100000
            best_play = -1
            for move in self.get_all_moves():
                modified = self.copy()
                m = deepcopy(move)
                m[0].board = modified
                modified.move(m)
                value, m = modified.minimax(ply_left - 1, alpha, beta)
                if value > current_max:
                    current_max = value
                    move[0].board = self
                    best_play = move
                    if alpha < current_max:
                        alpha = current_max
                    if beta <= alpha:
                        break
            return current_max, best_play
        else:  # MINI
            current_min = 100000
            best_play = None
            for move in self.get_all_moves():
                modified = self.copy()
                m = deepcopy(move)
                m[0].board = modified
                modified.move(m)
                value, m = modified.minimax(ply_left - 1, alpha, beta)
                if value < current_min:
                    current_min = value
                    best_play = move
                    if beta > current_min:
                        beta = current_min
                    if beta <= alpha:
                        break
            return current_min, best_play

    def make_random_move(self):
        all_moves = self.get_all_moves()
        move = choice(all_moves)
        p, pos = move
        move_copy = deepcopy(p), pos
        self.move(move)
        return move_copy

    def switch_team(self):
        self.active_team = WHITE if self.active_team == BLACK else BLACK

    def get_default_board(self):
        board = []
        base_row = [Rook, Knight, Bishop, Queen, King, Bishop, Knight, Rook]
        for r in range(self.board_size):
            row = []
            for c in range(self.board_size):
                if r == 0: row.append(base_row[c]((r, c), BLACK, self))
                elif r == 1: row.append(Pawn((r, c), BLACK, self))
                elif r == 6: row.append(Pawn((r, c), WHITE, self))
                elif r == 7: row.append(base_row[c]((r, c), WHITE, self))
                else: row.append(None)
            board.append(row)
        return board

    def get_all_moves(self):
        moves = []
        for row in self.board:
            for square in row:
                if square != None:
                    moves.extend(square.get_all_valid_moves())
        return moves

    def display(self):
        global window
        if not pygame_imported:
            import_pygame()
        step = min(window.get_size()) / self.board_size
        for row in range(self.board_size):
            for col in range(self.board_size):
                x, y = col * step, row * step
                pygame.draw.rect(window, WHITE if (row+col)%2==0 else BLACK, (x, y, step, step))
                piece = self.board[row][col]
                if piece is not None:
                    piece.draw(x, y)
        pygame.display.flip()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit()

    def draw_valid_moves(self):
        if not pygame_imported:
            import_pygame()
        for piece, moveTo in self.get_all_moves():
            pygame.draw.line(window, GREEN, self.get_square_center(*piece.pos), self.get_square_center(*moveTo))
        pygame.display.flip()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit()

    def on_board(self, r, c):
        return 0 <= r < self.board_size and 0 <= c < self.board_size

    def screen_to_square(self):
        step = min(window.get_size()) / self.board_size

    def get_square_center(self, r, c):
        step = min(window.get_size()) / self.board_size
        return (c+0.5) * step, (r+0.5) * step

    def move(self, move):
        piece, moveTo = move
        r1, c1 = piece.pos
        self.board[r1][c1] = None
        r2, c2 = moveTo
        self.board[r2][c2] = piece
        piece.pos = moveTo
        self.switch_team()

    def copy(self):
        new = deepcopy(self)
        for r in range(self.board_size):
            for c in range(self.board_size):
                val = new.board[r][c]
                if val is not None:
                    new.board[r][c] = type(val)(val.pos, val.team, new)
        return new

    def __repr__(self):
        return f"Board: {self.id}"


class ChessPiece:
    val = 0
    def __init__(self, pos, team, board):
        self.board = board
        self.pos = pos
        self.team = team

    def draw(self, x, y):
        x, y = round(x), round(y)
        # pygame.draw.circle(window, RED, (x, y), 5)
        message = type(self).__name__
        text_color = RED if self.team == WHITE else BLUE  # (0, 255, 0)
        text_font = 'Comic Sans MS'
        text_size = 25
        font = pygame.font.SysFont(text_font, text_size)
        score_surface = font.render(message, False, text_color)
        window.blit(score_surface, (x, y+10))

    def get_all_valid_moves(self):
        return []

    def __repr__(self):
        return f"{'white' if self.team == WHITE else 'black'} {type(self).__name__} @ {self.pos}"


class Pawn(ChessPiece):
    val = 1
    def get_all_valid_moves(self):
        if self.board.active_team != self.team: return []
        moves = []
        r, c = self.pos
        start = 6 if self.team == WHITE else 1
        front = r-1 if self.team == WHITE else r+1
        front2 = r-2 if self.team == WHITE else r+2
        # normal push
        if self.board.on_board(front, c) and self.board.board[front][c] is None:
            moves.append((self, (front, c)))
            # double push
            if r == start and self.board.board[front2][c] is None:
                moves.append((self, (front2, c)))
        # taking
        if self.board.on_board(front, c-1):
            piece = self.board.board[front][c-1]
            if piece is not None and piece.team != self.team:
                moves.append((self, (front, c-1)))
        if self.board.on_board(front, c + 1):
            piece = self.board.board[front][c + 1]
            if piece is not None and piece.team != self.team:
                moves.append((self, (front, c + 1)))
        return moves


class Knight(ChessPiece):
    val = 3
    transforms = [(2, 1), (1, 2), (-2, 1), (-1, 2), (2, -1), (1, -2), (-2, -1), (-1, -2)]

    def get_all_valid_moves(self):
        if self.board.active_team != self.team: return []
        moves = []
        r, c = self.pos
        for dc, dr in self.transforms:
            new_r, new_c = r+dr, c+dc
            if self.board.on_board(new_r, new_c):
                piece = self.board.board[new_r][new_c]
                if piece is None or piece.team != self.team:
                    moves.append((self, (new_r, new_c)))
        return moves


class Bishop(ChessPiece):
    val = 3

    def get_all_valid_moves(self):
        if self.board.active_team != self.team: return []
        r, c = self.pos
        moves = []
        for dr in [-1, 1]:
            for dc in [-1, 1]:
                new_r, new_c = r + dr, c + dc
                while self.board.on_board(new_r, new_c):
                    piece = self.board.board[new_r][new_c]
                    if piece is None or piece.team != self.team:
                        moves.append((self, (new_r, new_c)))
                        new_r += dr
                        new_c += dc
                    if piece is not None:
                        break
        return moves


class Rook(ChessPiece):
    val = 5
    transforms = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    def get_all_valid_moves(self):
        if self.board.active_team != self.team: return []
        r, c = self.pos
        moves = []
        for dr, dc in Rook.transforms:
            new_r, new_c = r + dr, c + dc
            while self.board.on_board(new_r, new_c):
                piece = self.board.board[new_r][new_c]
                if piece is None or piece.team != self.team:
                    moves.append((self, (new_r, new_c)))
                    new_r += dr
                    new_c += dc
                if piece is not None:
                    break
        return moves


class Queen(ChessPiece):
    val = 9
    def get_all_valid_moves(self):
        if self.board.active_team != self.team: return []
        moves = []
        moves.extend(Rook.get_all_valid_moves(self))
        moves.extend(Bishop.get_all_valid_moves(self))
        return moves


class King(ChessPiece):
    val = 200

    def get_all_valid_moves(self):
        if self.board.active_team != self.team: return []
        r, c = self.pos
        moves = []
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                new_r, new_c = r + dr, c + dc
                if self.board.on_board(new_r, new_c):
                    piece = self.board.board[new_r][new_c]
                    if piece is None or piece.team != self.team:
                        moves.append((self, (new_r, new_c)))
        return moves


if __name__ == '__main__':
    c = ChessPosition()
    c.play_minimax(True)