import pdb
from helpers import normalize, blur

def initialize_beliefs(grid):
    height = len(grid)
    width = len(grid[0])
    area = height * width
    belief_per_cell = 1.0 / area
    beliefs = []
    for i in range(height):
        row = []
        for j in range(width):
            row.append(belief_per_cell)
        beliefs.append(row)
    return beliefs

def sense(color, grid, beliefs, p_hit, p_miss):
    new_beliefs = []

    #
    # TODO - implement this in part 2
    #
    expect = []
    total_sum = 0
    for b_row, g_row in zip(beliefs, grid):
        expect_row = []
        running_sum = 0
        for b_cell, g_cell in zip(b_row, g_row):
            hit = (color == g_cell)
            expect_row.append(b_cell * (hit * p_hit + (1-hit) * p_miss ) )
            running_sum = sum(expect_row)
        expect.append(expect_row)
        total_sum = total_sum + running_sum

    for row in range(len(expect)):
        for cell in range(len(expect[row])):
            expect[row][cell] = expect[row][cell] / total_sum
    return expect

def move(dy, dx, beliefs, blurring):
    height = len(beliefs)
    width = len(beliefs[0])
    # print(height, width)
    new_G = [[0.0 for i in range(width)] for j in range(height)]
    for i, row in enumerate(beliefs):
        for j, cell in enumerate(row):
            new_i = (i + dy ) % height
            new_j = (j + dx ) % width
            # print(new_i, new_j)
            #pdb.set_trace()
            new_G[int(new_i)][int(new_j)] = cell
    return blur(new_G, blurring)