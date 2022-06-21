from Cards.General import *


def play_random(print_=False):
    d = Deck()
    d.shuffle()

    # color
    color = random.choice(["Red", "Black"])
    card1 = d.draw()
    red = card1.suit in ["Diamonds", "Hearts"]
    if not (color == "Red") == red:
        if print_: print("red: ", red, card1)
        return False

    card2 = d.draw()
    over = random.choice([True, False])
    if not (card1.value > card2.value) == over:
        if print_: print("over: ", red, card1, card2)
        return False

    card3 = d.draw()
    between = random.choice([True, False])
    if not (card1.value < card3.value < card2.value or card2.value < card3.value < card1.value) == over:
        if print_: print("between: ", between, card1, card2, card3)
        return False

    card4 = d.draw()
    suit = random.choice(["Diamonds", "Hearts", "Spades", "Clubs"])
    if print_: print("suit: ", suit, card4.suit)
    return suit == card4.suit


def play_good(print_=False):
    d = Deck()
    d.shuffle()

    # color
    color = random.choice(["Red", "Black"])
    card1 = d.draw()
    red = card1.suit in ["Diamonds", "Hearts"]
    if not (color == "Red") == red:
        if print_: print("red: ", red, card1)
        return False

    card2 = d.draw()
    over = card1.value > 9
    if not (card1.value > card2.value) == over:
        if print_: print("over: ", red, card1, card2)
        return False

    card3 = d.draw()
    between = abs(card1.value - card2.value) < 9
    if card3.value in [card2.value or card1.value] or \
            not (card1.value < card3.value < card2.value or card2.value < card3.value < card1.value) == over:
        if print_: print("between: ", between, card1, card2, card3)
        return False

    card4 = d.draw()
    suits = ["Diamonds", "Hearts", "Spades", "Clubs"]
    for card in (card1, card2, card3):
        if card.suit in suits:
            suits.remove(card.suit)
    suit = random.choice(suits)
    if print_: print("suit: ", suit, card4.suit)
    return suit == card4.suit


def odds(func, count=10000):
    results = [func() for i in range(count)]
    return results.count(True) / count


if __name__ == '__main__':
    # play_random()
    # print(odds(play_random))
    print(odds(play_good))