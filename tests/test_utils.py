from unittest.mock import patch

import numpy as np
import pytest

from scripts.utils import (check_basic_win_condition_for_blue_player, check_basic_win_condition_for_red_player,
                           encode_position, get_hex_neighbourhood, get_winner)
from scripts.utils import negamax_alpha_beta_pruned_with_transposition_tables as negamax


# a1  b1  c1
#  a2  b2  c2
#   a3  b3  c3

# a1  b1  c1  d1
#  a2  b2  c2  d2
#   a3  b3  c3  d3
#    a4  b4  c4  d4


@pytest.mark.parametrize("list_of_moves, size, is_fulfilled", [
    ([], 1, False),
    ([], 5, False),
    (['a1'], 1, True),
    (['a1'], 2, False),
    (['a1', 'a2'], 2, False),
    (['a1', 'b1'], 2, True),
    (['a1', 'b2', 'c3', 'd2', 'e1'], 5, True),
    (['a1', 'b2', 'c3', 'd2', 'd3'], 5, False),
])
def test_check_basic_win_condition_for_blue_player(list_of_moves, size, is_fulfilled):
    list_of_moves = list(map(encode_position, list_of_moves))
    assert check_basic_win_condition_for_blue_player(list_of_moves, size) == is_fulfilled


@pytest.mark.parametrize("list_of_moves, size, is_fulfilled", [
    ([], 1, False),
    ([], 5, False),
    (['a1'], 1, True),
    (['a1'], 2, False),
    (['a1', 'b1'], 2, False),
    (['a1', 'a2'], 2, True),
    (['a1', 'b2', 'c3', 'b4', 'a5'], 5, True),
    (['a1', 'b2', 'c3', 'b4', 'a4'], 5, False),
])
def test_check_basic_win_condition_for_red_player(list_of_moves, size, is_fulfilled):
    list_of_moves = list(map(encode_position, list_of_moves))
    assert check_basic_win_condition_for_red_player(list_of_moves, size) == is_fulfilled


@pytest.mark.parametrize("position, neighbours", [
    ('b2', ['b1', 'c1', 'c2', 'b3', 'a3', 'a2']),
])
def test_get_hex_neighbourhood(position, neighbours):
    position = encode_position(position)
    neighbours = list(map(encode_position, neighbours))
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
    ),
    (
        [],
        ['c1'],
        'c1',
        3,
        None
    ),
])
def test_get_winner(red_moves, blue_moves, last_move, size, winner):
    red_moves = list(map(encode_position, red_moves))
    blue_moves = list(map(encode_position, blue_moves))
    last_move = encode_position(last_move)
    assert get_winner(red_moves, blue_moves, last_move, size) == winner


@patch.dict('scripts.utils.TRANSPOSITION_TABLE', {})
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
def test_negamax_predicts_score(player, red_moves, blue_moves, last_move, size, score):
    red_moves = list(map(encode_position, red_moves))
    blue_moves = list(map(encode_position, blue_moves))
    last_move = encode_position(last_move) if last_move else tuple()
    alpha, beta = -np.inf, np.inf
    assert negamax(player, red_moves, blue_moves, last_move, alpha, beta, size)['score'] == score


@patch.dict('scripts.utils.TRANSPOSITION_TABLE', {})
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
        ['c1', 'b3'],
        ['c2', 'c3', 'a2'],
        'a2',
        4,
        (1, 1)
    ),
])
def test_negamax_predicts_move(player, red_moves, blue_moves, last_move, size, move):
    red_moves = list(map(encode_position, red_moves))
    blue_moves = list(map(encode_position, blue_moves))
    last_move = encode_position(last_move)
    alpha, beta = -np.inf, np.inf
    assert negamax(player, red_moves, blue_moves, last_move, alpha, beta, size)['move'] == move
