import numpy as np
import pytest

from scripts.utils import (get_hex_neighbourhood, check_if_connects_to_nth_row, check_if_connects_to_nth_column,
                           get_winner, negamax_alpha_beta_pruned)


# a1  b1  c1
#  a2  b2  c2
#   a3  b3  c3

# a1  b1  c1  d1
#  a2  b2  c2  d2
#   a3  b3  c3  d3
#    a4  b4  c4  d4

@pytest.mark.parametrize("position, neighbours", [
    ('b2', ['b1', 'c1', 'c2', 'b3', 'a3', 'a2']),
])
def test_get_hex_neighbourhood(position, neighbours):
    assert set(get_hex_neighbourhood(position)) == set(neighbours)


@pytest.mark.parametrize("red_moves, blue_moves, last_move, size, winner", [
    (
        ['b1', 'b2', 'a1', 'a3'],
        ['c1', 'c2', 'a2'],
        'a3',
        3,
        1
    ),
    (
        ['a1', 'b1'],
        ['c1', 'b2', 'a2'],
        'a2',
        3,
        -1
    ),
    (
        ['a1', 'b1'],
        ['c1'],
        'a1',
        3,
        None
    )
])
def test_get_winner(red_moves, blue_moves, last_move, size, winner):
    assert get_winner(red_moves, blue_moves, last_move, size) == winner


@pytest.mark.parametrize("player, red_moves, blue_moves, last_move, size, score", [
    (
        1,
        ['b1', 'b2'],
        ['c1', 'c2'],
        'c2',
        3,
        1
    ),
    (
        -1,
        ['b1', 'b2', 'a1'],
        ['c1', 'c2'],
        'a1',
        3,
        -1
    ),
    (
        1,
        ['b1', 'b2', 'a1'],
        ['c1', 'c2', 'a2'],
        'a2',
        3,
        1
    ),
    (
        -1,
        ['b1', 'b2', 'a1', 'a3'],
        ['c1', 'c2', 'a2'],
        'a3',
        3,
        -1
    ),
    (
        1,
        ['b1', 'b2', 'a1'],
        ['c1', 'c2', 'b3'],
        'b3',
        3,
        1
    ),
    (
        -1,
        ['b1', 'b2', 'a1'],
        ['c1', 'c2', 'b3'],
        'a1',
        3,
        1
    ),
    (
        1,
        ['a1', 'b1'],
        ['c1', 'c2', 'b3'],
        'b3',
        3,
        1
    ),
    (
        -1,
        ['a1', 'b1'],
        ['c1'],
        'a1',
        3,
        1
    ),
    (
        -1,
        ['a1'],
        [],
        'a1',
        3,
        1
    ),
    (
        1,
        [],
        [],
        '',
        3,
        1
    ),
    (
        1,
        ['c1'],
        ['b2'],
        'b2',
        3,
        1
    ),
    (
        1,
        ['c1', 'a2'],
        ['b2', 'a3'],
        'a3',
        3,
        1
    ),
    (
        1,
        ['c1', 'a2', 'c2'],
        ['b2', 'a3', 'b3'],
        'b3',
        3,
        1
    ),
    (
        1,
        ['c1', 'a2'],
        ['b2', 'a3'],
        'a3',
        4,
        1
    ),
])
def test_negamax_alpha_beta_pruned_predicts_score(player, red_moves, blue_moves, last_move, size, score):
    alpha, beta = -np.inf, np.inf
    assert negamax_alpha_beta_pruned(player, red_moves, blue_moves, last_move, alpha, beta, size)['score'] == score


@pytest.mark.parametrize("player, red_moves, blue_moves, last_move, size, move", [
    (
        1,
        ['b1', 'a3'],
        ['c2', 'b2'],
        'b2',
        3,
        (1, 0)
    ),
    (
        1,
        ['b1', 'a3'],
        ['c2', 'b2'],
        'b2',
        4,
        (1, 0)
    ),
])
def test_negamax_alpha_beta_pruned_predicts_move(player, red_moves, blue_moves, last_move, size, move):
    alpha, beta = -np.inf, np.inf
    assert negamax_alpha_beta_pruned(player, red_moves, blue_moves, last_move, alpha, beta, size)['move'] == move
