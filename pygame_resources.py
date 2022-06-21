from time import time
from math import *
import pygame
pygame.init()

RED = (255, 0, 0)
LIGHT_RED = (255, 50, 50)
YELLOW = (255, 255, 0)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
LIGHT_BROWN = (165, 42, 42)
WHITE = (255, 255, 255)
GREY = (100, 100, 100)
LIGHT_GREY = (150, 150, 150)
OCEAN_BLUE = (0, 0, 155)
LIME_GREEN = (127, 255, 0)
GREEN = (0, 255, 0)
DARK_GREEN = (0, 100, 0)
PINK = (255, 192, 203)
ORANGE = (255, 69, 0)
BROWN = (165, 42, 42)
LIGHT_BLUE = (173, 216, 230)


def freeze_display(t=1000):
    start_time = time()
    while time() - start_time < t:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit()


def towards(pos1, pos2):
    x1, y1 = pos1
    x2, y2 = pos2
    dx = x2 - x1
    dy = y2 - y1
    angle = atan(dy / dx) if dx != 0 else pi/2 if dy > 0 else 3*pi/2
    angle += pi if dx < 0 else 0
    return angle


def get_components(angle, hypo):
    return cos(angle) * hypo, sin(angle) * hypo


def distance(pos1, pos2):
    return sqrt(sum((x1 - x2)**2 for x1, x2 in zip(pos1, pos2)))


def get_unit_vector(vector):
    magnitude = vector_magnitude(vector)
    unit_vector = tuple(component / magnitude for component in vector)
    return unit_vector


def resize_vector(vector, length):
    unit_vector = get_unit_vector(vector)
    return tuple(component * length for component in unit_vector)


def shorten_vector(vector, max):
    magnitude = vector_magnitude(vector)
    return vector if magnitude <= max else resize_vector(vector, max)


def vector_magnitude(vector):
    return distance((0 for i in range(len(vector))), vector)


def vector_angle(vector):
    return towards((0 for i in range(len(vector))), vector)


def reflect(vector, angle):
    income_rotation = vector_angle(vector)
    output_angle = 2 * angle - income_rotation
    return get_components(radians(output_angle), vector_magnitude(vector))


def check_events():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            quit()


def display(window, msg, cords, size=25, color=BLACK, font='timesnewromanttf'):
    msg = str(msg)
    text_color = color
    text_font = font
    text_size = size
    font = pygame.font.SysFont(text_font, text_size)
    score_surface = font.render(msg, False, text_color)

    window.blit(score_surface, cords)


def draw_function(f, domain, range, origin, width, height, window, color=BLUE):
    if width == 0 or height == 0: return
    start_x, end_x = domain
    domain_size = end_x - start_x
    low_y, top_y = range
    range_size = top_y-low_y
    step = domain_size/width
    upstep = range_size/height
    ox, oy = origin
    x = start_x
    while x <= end_x:
        y = f(x)
        # print(x, y)
        if not low_y < y < top_y:
            pass
        else:
            dx = (x-start_x)/step
            dy = y/upstep * -1  # because pygame coordinates are weird
            pos = round(ox + dx), round(oy + dy)
            # print(pos)
            pygame.draw.line(window, color, pos, pos)
        x += step


def round_tup(tup):
    return tuple(round(x) for x in tup)

