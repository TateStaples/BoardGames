from Misc.Aaron_game.Towers import Tower
from Misc.Aaron_game.Tiles import Castle


class Troop:
    cost = 0
    max_health = 500
    attack_damage = 10
    targeting = "all"
    movement_range = 5
    attack_range = 1
    price = 100

    def __init__(self, player):
        self.player = player
        self.player.troops.append(self)
        self.health = self.max_health
        self.moved = False

    def valid_target(self, target):
        if target is None or not target.player.alive:
            return True
        if self.targeting == "all":
            return True
        if self.targeting == "tower":
            return isinstance(target, Tower) or isinstance(target, Castle)
        if self.targeting == "team":
            return isinstance(target, Troop) and target.player is self.player
        return False

    def attack(self, target):
        return target.damage(self.attack_damage)

    def damage(self, amount):
        self.health -= amount
        return self.health < 0


class Farmer(Troop):
    targeting = None


class PigRider(Troop):
    targeting = "tower"


class Magician(Troop):
    targeting = "all"


class Medic(Troop):
    targeting = "team"
    attack_damage = -50


class Viking(Troop):
    attack_damage = 200


class Robot(Troop):
    max_health = 3000
    movement_range = 20
    attack_damage = 300
