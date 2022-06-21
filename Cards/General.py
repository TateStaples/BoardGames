import random


class Deck:
    deck_size = 52
    def __init__(self):
        self.available = [Card(i) for i in range(self.deck_size)]
        self.in_use = list()

    def draw(self):
        card = self.available.pop(0)
        self.in_use.append(card)
        return card

    def shuffle(self):
        random.shuffle(self.available)

    @property
    def top_card(self):
        return self.available[0]


class Card:
    suits = ['Hearts', 'Diamonds', 'Spades', "Clubs"]
    card_amount = 13

    def __init__(self, raw):
        self._raw = raw

    @property
    def value(self):
        return self._raw % self.card_amount + 2

    @property
    def suit(self):
        return self.suits[self._raw // self.card_amount]

    def __str__(self):
        face = {11: "Jack", 12: "Queen", 13: "King", 14: "Ace"}
        return f"{self.value if self.value <=10 else face[self.value]} of {self.suit}"