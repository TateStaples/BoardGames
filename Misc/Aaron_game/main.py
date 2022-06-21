import pygame_resources
import pygame
from Misc.Aaron_game import Tiles
from Misc.Aaron_game import Troops
from Misc.Aaron_game import Towers
from Misc.Aaron_game import Spells


class Player:
    players = []
    sole_winner = False

    def __init__(self):
        self.players.append(self)
        self.castle = None
        self.dollars = None
        self.gold = None
        self.shop = [[Troops.PigRider, Troops.Magician, Troops.Robot, Troops.Farmer, Troops.Medic],
                     [Towers.Canon, Towers.X_bow], [Spells.Rocket]]
        self.troops = []
        self.towers = []
        self.alive = True

    def establish_castle(self, _class):
        index = len(self.players)-1
        locs = [(0, 0), (0, Game.board_width-1), (Game.board_height-1, 0), (Game.board_height-1, Game.board_width-1)]
        self.castle = _class(Game.colors[index], locs[index], self)
        self.gold = self.castle.start_money
        self.dollars = self.castle.start_money

    def earn_income(self):
        self.dollars += self.castle.income
        self.gold += self.castle.income

    def end_turn(self):
        for tower in self.towers:
            tower.moved = False
        for troop in self.troops:
            troop.moved = False
        self.earn_income()

    def die(self, killer):
        self.alive = False
        for r in range(Game.board_height):
            for c in range(Game.board_width):
                spot = game.board[r][c]
                if spot is not None and not isinstance(spot, Tiles.Castle):
                    if spot.player is self:
                        game.boom(r, c)
        assert isinstance(killer, Player)
        killer.dollars += self.dollars // 2
        killer.gold += self.gold // 2
        # Game.colors[self.players.index(self)] = pygame_resources.GREY
        self.castle.color = pygame_resources.GREY
        count = 0
        for player in self.players:
            if player.alive:
                count += 1
        self.sole_winner = count == 1
        if count < 1: raise Exception("Everybody dead. How'd that happen?")



class Game:
    board_width = 20
    board_height = 20
    board = []
    colors = [pygame_resources.RED, pygame_resources.BLUE, pygame_resources.ORANGE, pygame_resources.DARK_GREEN]
    selected_square = None
    player_index = 0
    small_farm_tiles = []
    large_farm_tiles = []

    def __init__(self, num_players):
        self.num_players = num_players
        for i in range(num_players):
            castle = self.get_castle(i)
            p = Player()
            p.establish_castle(castle)

    def play(self):
        while True:
            self.check_events()

    def setup_board(self):
        # a blank board
        for r in range(self.board_width):
            row = []
            for c in range(self.board_width):
                row.append(None)
            self.board.append(row)

        # add castles
        for p in Player.players:
            r, c = p.castle.location
            for dr in range(-p.castle.size, p.castle.size + 1):
                for dc in range(-p.castle.size, p.castle.size + 1):
                    nr, nc = r + dr, c + dc
                    if self.in_board(nr, nc):
                        self.board[nr][nc] = p.castle

        # add farms
        h = Game.board_height
        w = Game.board_width
        small_size = 3
        half = small_size // 2
        small = [(w//2, h//4), (w//4, h//2), (w//2, h//4*3), (w//4*3, h//2)]
        for x, y in small:
            for dx in range(-half, half+1):
                for dy in range(-half, half+1):
                    self.small_farm_tiles.append((x+dx, y+dy))
        half = 2
        x, y = h//2, w//2
        for dx in range(-half, half + 1):
            for dy in range(-half, half + 1):
                self.large_farm_tiles.append((x+dx, y+dy))

        self.board[5][5] = Troops.PigRider(self.get_player())

    def get_castle(self, num):
        return Tiles.Demo
        castles = {"rich": Tiles.Rich, "medic": Tiles.Medic, "farmer": Tiles.Farmer}
        answer = ""
        while answer not in castles:
            answer = input(f"Choose Castle types for Player {num+1} {list(castles.keys())}:  ").lower()
        return castles[answer]

    def get_troops(self):
        pass

    def check_events(self):
        for event in pygame.event.get():
            if event == pygame.QUIT:
                quit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                clicked = pygame.mouse.get_pos()
                self.click(*clicked)

    def click(self, x, y):
        # check if next
        nx, ny, w, h = Window.next
        if nx < x < nx + w and ny < y < ny + h:
            print("next")
            self.next_player()
            return

        box = Window.get_box(x, y)
        if self.selected_square is None:  # nothing selected before
            if box is None:  # click outside of board
                # check shop
                troop = self.get_shop_item(x, y)
                self.selected_square = troop
            else:
                self.selected_square = box
            self.illuminate_actions()
            print(self.selected_square)
            print(self.get_available_actions())
        else:  # this is the second click
            self.do_action(box)

    def do_action(self, box):
        actions = self.get_available_actions()
        if actions is None or box not in self.get_available_actions():
            self.selected_square = None
            Window.draw_all()
            return
        r, c = box
        if isinstance(self.selected_square, tuple):  # moving or attacking
            pr, pc = self.selected_square
            acting_troop = self.board[pr][pc]
            target = self.board[r][c]
            if isinstance(acting_troop, Tiles.Castle):  # attacking with castle
                if acting_troop.attack(target):
                    self.boom(r, c)
            elif isinstance(acting_troop, Troops.Troop):
                if target is None:  # move
                    self.board[pr][pc] = None
                    self.board[r][c] = acting_troop
                else:
                    if acting_troop.attack(target):
                        self.boom(r, c)
            else:  # tower
                if acting_troop.attack(target):
                    self.boom(r, c)
            acting_troop.moved = True

        elif issubclass(self.selected_square, Troops.Troop) or issubclass(self.selected_square, Towers.Tower):
            thing = self.selected_square(self.get_player())
            self.board[r][c] = thing
            thing.moved = True
            self.get_player().gold -= thing.price
        elif issubclass(self.selected_square, Spells.Spell):
            target = self.board[r][c]
            if target.damage(self.selected_square.damage):
                self.board[r][c]
            self.get_player().dollars -= self.selected_square.price

        Window.draw_all()
        self.selected_square = None

    def illuminate_actions(self):
        actions = self.get_available_actions()
        if actions is None:
            return
        for r, c in actions:
            troop_at = self.board[r][c]
            color = pygame_resources.YELLOW if troop_at is None else pygame_resources.RED
            pygame.draw.rect(Window.surface, color, (*Window.get_box_coord(r, c), Window.box_width, Window.box_height), 2)
            Window.update_square(r, c)

    def get_available_actions(self):
        if self.selected_square is None:
            return None
        if isinstance(self.selected_square, tuple):  #
            r, c = self.selected_square
            troop = self.board[r][c]
            if troop is None or troop.moved:
                return None
            elif isinstance(troop, Tiles.Castle):
                if self.get_player().castle != troop:
                    return None
                r, c = troop.location
                speed = Tiles.Castle.size + Tiles.Castle.attack_range
            elif isinstance(troop, Towers.Tower):
                speed = troop.range
            else:
                speed = troop.movement_range
            available_spots = set()
            for dr in range(-speed, speed+1):
                for dc in range(-(speed-abs(dr)), speed-abs(dr)):
                    nr, nc = r + dr, c + dc
                    if (dr, dc) == (0, 0) or not self.in_board(nr, nc): continue
                    troop_at_spot = self.board[nr][nc]
                    if troop.valid_target(troop_at_spot):
                        available_spots.add((nr, nc))
            return available_spots
        elif isinstance(self.selected_square, Spells.Spell):  # can go anywhere
            spell = self.selected_square
            if self.get_player().gold < spell.price:
                return None
            return set((r, c) for r, c in zip(range(self.board_height), range(self.board_width)))
        elif issubclass(self.selected_square, Towers.Tower) or issubclass(self.selected_square, Troops.Troop):
            tower = self.selected_square
            if self.get_player().dollars < tower.price:
                return None
            r, c = self.get_player().castle.location
            radius = Tiles.Castle.size + Tiles.Castle.spawn_range
            available_locations = []
            for dr in range(-radius, radius+1):
                for dc in range(-radius, radius+1):
                    nr, nc = r+dr, c+dc
                    if self.in_board(nc, nr) and self.board[nr][nc] is None:
                        available_locations.append((nr, dc))
            return available_locations

    @staticmethod
    def in_board(r, c):
        return 0 <= r < Game.board_height and 0 <= c < Game.board_width

    def next_player(self):
        self.get_player().end_turn()
        if not Player.sole_winner:
            self.player_index = (self.player_index + 1) % self.num_players
            while not self.get_player().alive:
                self.player_index = (self.player_index + 1) % self.num_players
        else:
            index = 0
            for i, p in enumerate(Player.players):
                if p.alive:
                    index = i+1
                    break
            pygame_resources.display(Window.surface, f"Player {index} has won.", (10, Window.window_height), size=50)
            pygame_resources.freeze_display()
            quit()
        self.selected_square = None
        Window.draw_all()

    def get_player(self):
        return Player.players[self.player_index]

    def get_shop_item(self, click_x, click_y):
        for shop_section, location in enumerate([Window.troop_title, Window.tower_title, Window.spell_title]):
            for section_index, troop in enumerate(game.get_player().shop[shop_section]):
                x, y = location
                new_y = y + section_index * Window.shop_box_size + 30
                if x < click_x < x + Window.shop_box_size and new_y < click_y < new_y + Window.shop_box_size:
                    return troop
        return None

    def boom(self, r, c):
        thing = self.board[r][c]
        if thing is None:
            print("bad boom " + str((r, c)))
        elif isinstance(thing, Tiles.Castle):
            thing.player.die(self.get_player())
        else:
            self.board[r][c] = None


class Window:
    window_width = 600
    window_height = 800
    info_height = 200

    box_width = window_width // Game.board_width
    box_height = (window_height-info_height) // Game.board_width

    surface = pygame.display.set_mode((window_width, window_height))

    name_location = 10, window_height-info_height + 10
    health_info = name_location[0], name_location[1] + 40
    dollars_info = name_location[0], name_location[1] + 60
    gold_info = name_location[0], name_location[1] + 80
    title_shop_size = 25
    info_size = 15
    shop_box_size = 20
    troop_title = (window_width // 10 * 4, window_height-info_height + info_height//5)
    tower_title = (window_width // 10 * 6, window_height-info_height + info_height//5)
    spell_title = (window_width // 10 * 8, window_height-info_height + info_height//5)

    next = (name_location[0], name_location[1] + 120, 50, 20)

    @staticmethod
    def get_box(x, y):
        row = y // Window.box_height
        col = x // Window.box_width
        if not Game.in_board(row, col):
            return None
        return row, col

    @staticmethod
    def get_box_coord(r, c):
        if r >= Game.board_height or c >= Game.board_width:
            raise Exception(f"attempted corodinate {(r, c)} does not exist")
        return c * Window.box_width, r * Window.box_height

    @staticmethod
    def draw_all():
        Window.clear()
        Window.draw_game()
        Window.draw_info()
        pygame.display.update()

    @staticmethod
    def draw_game():
        for r, row in enumerate(Game.board):
            for c, tile in enumerate(row):
                if (r, c) in game.large_farm_tiles or (r, c) in game.small_farm_tiles:
                    color = pygame_resources.GREEN
                elif tile is None or isinstance(tile, Troops.Troop) or isinstance(tile, Towers.Tower):
                    color = pygame_resources.LIGHT_BROWN
                elif type(tile) == int:
                    color = Game.colors[tile]
                else:
                    color = tile.color
                x, y = Window.get_box_coord(r, c)
                if isinstance(tile, Tiles.Castle) or type(tile) == int:
                    pygame.draw.rect(Window.surface, color, (x, y, Window.box_width, Window.box_height))
                else:
                    pygame.draw.rect(Window.surface, color, (x, y, Window.box_width-1, Window.box_height-1))
                Window.draw_troop(x, y, tile)

    @staticmethod
    def draw_troop(x, y, troop):
        if isinstance(troop, Troops.Troop) or isinstance(troop, Towers.Tower):
            initial = type(troop).__name__ if type(troop) != type else troop.__name__
            color = Game.colors[Player.players.index(troop.player)]
            msg = f"{troop.health}/{troop.max_health}"
            pygame_resources.display(Window.surface, initial[0], (x, y), size=15, color=color)
            pygame_resources.display(Window.surface, msg, (x, y+Window.box_width//5*3), size=10, color=color)

    @staticmethod
    def draw_info():
        # stats
        player = game.get_player()
        pygame_resources.display(Window.surface, f"Player {game.player_index+1}:", Window.name_location, 30)
        pygame_resources.display(Window.surface, f"Tower health = {player.castle.max_health}", Window.health_info, Window.info_size)
        pygame_resources.display(Window.surface, f"Amount of gold = {player.dollars}", Window.gold_info, Window.info_size)
        pygame_resources.display(Window.surface, f"Amount of dollars = {player.gold}", Window.dollars_info, Window.info_size)

        # shop
        for shop_section, (name, location) in enumerate([("Troops",  Window.troop_title), ("Towers", Window.tower_title), ("Spells", Window.spell_title)]):
            pygame_resources.display(Window.surface, name, location, Window.title_shop_size)
            for section_index, troop in enumerate(player.shop[shop_section]):
                x, y = location
                new_y = y + section_index * Window.shop_box_size + 30
                new_x = x + 20
                pygame.draw.rect(Window.surface, pygame_resources.YELLOW, (x, new_y, Window.shop_box_size//2, Window.shop_box_size//2))
                initial = type(troop).__name__ if type(troop) != type else troop.__name__
                pygame_resources.display(Window.surface, initial, (new_x, new_y), size=Window.info_size)
                # Window.draw_troop(x, new_y, game.get_player().shop[i])

        # next button
        x, y, w, h = Window.next
        pygame.draw.rect(Window.surface, (0, 0, 0), Window.next, 1)
        pygame_resources.display(Window.surface, "Next", (x+w//10, y+h//10), size=15)

        # castle healths
        for player in Player.players:
            castle = player.castle
            r, c = castle.location

            x, y = Window.get_box_coord(min(r, Game.board_height-4), min(c, Game.board_width-4),)
            pygame_resources.display(Window.surface, f"{castle.health}/{castle.max_health}", (x, y), size=10)

    @staticmethod
    def clear():
        Window.surface.fill(pygame_resources.WHITE)

    @staticmethod
    def update_square(r, c):
        x, y = Window.get_box_coord(r, c)
        pygame.display.update((x, y, Window.box_width, Window.box_height))



if __name__ == '__main__':
    game = Game(4)
    game.setup_board()
    Window.draw_all()
    game.play()
    # pygame_resources.freeze_display(100)