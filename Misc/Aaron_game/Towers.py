class Tower:
    range = None
    max_health = None
    attack_damage = 100
    price = 100

    def __init__(self, player):
        self.health = self.max_health
        self.moved = False
        self.player = player

    def valid_target(self, troop):
        pass

    def damage(self, amount):
        self.health -= amount
        return self.health <= 0

    def attack(self, target):
        return target.damage(self.attack_damage)


class Canon(Tower):
    max_health = 300


class X_bow(Tower):
    max_health = 1000