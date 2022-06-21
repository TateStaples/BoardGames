from pygame_resources import *


class Tile:
    def __init__(self, color):
        self.color = color


class Castle(Tile):  # abstract class
    max_health = 0
    start_money = 0
    income = 0
    attack_damage = 100
    size = 4
    attack_range = 4
    spawn_range = 2

    def __init__(self, color, location, player):
        super(Castle, self).__init__(color)
        self.health = self.max_health
        self.location = location
        self.moved = False
        self.player = player
        print(location)

    def valid_target(self, target):
        return target is not None and target.player is not self.player

    def damage(self, amount):
        self.health -= amount
        return self.health <= 0

    def attack(self, target):
        return target.damage(self.attack_damage)


class Rich(Castle):
    max_health = 12000
    start_money = 750
    income = 75


class Medic(Castle):
    max_health = 16000
    start_money = 300
    income = 100


class Farmer(Castle):
    max_health = 11000
    start_money = 150
    income = 200


class Demo(Castle):
    max_health = 500
    start_money = 1000
    income = 10000


class Farm(Tile):
    farm_color = GREEN

    def __init__(self, income_value):
        super(Farm, self).__init__(self.farm_color)
        self.income_value = income_value