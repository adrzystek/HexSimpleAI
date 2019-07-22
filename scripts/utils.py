from typing import AbstractSet, Dict, Iterable, List, Optional, Tuple

import numpy as np


def encode_position(position: str) -> Tuple[int, int]:
    """
    Convert board coordinates from human-readable to computer-friendly.

    For example, 'a2' becomes (1, 0).
    """
    row = int(position[1:]) - 1
    col = ord(position[0]) - 96 - 1
    return row, col


def decode_position(position: Tuple[int, int]) -> str:
    """
    Convert board coordinates from computer-friendly to human-readable.

    For example, (1, 0) becomes 'a2'.
    """
    row, col = position
    new_position = chr(col + 1 + 96) + str(1 + row)
    return new_position


def get_hex_neighbourhood(position: Tuple[int, int]) -> List[Tuple[int, int]]:
    """
    Returns coordinates of adjacent hexes.

    Not all coordinates might be valid - in case of the `position` being a hex at the edge of the board,
    some returned positions will be invalid. This, however, is not a problem having taken the fact as this function
    is used.
    """
    row, col = position
    a = (row-1, col+1)
    b = (row, col+1)
    c = (row+1, col)
    d = (row+1, col-1)
    e = (row, col-1)
    f = (row-1, col)
    return [a, b, c, d, e, f]


def check_basic_win_condition_for_red_player(list_of_moves: Iterable[Tuple[int, int]], size: int) -> bool:
    rows = set([move[0] for move in list_of_moves])
    return len(rows) == size


def check_basic_win_condition_for_blue_player(list_of_moves: Iterable[Tuple[int, int]], size: int) -> bool:
    cols = set([move[1] for move in list_of_moves])
    return len(cols) == size


def check_if_connects_to_nth_row(move: Tuple[int, int], list_of_moves: Iterable[Tuple[int, int]], row_number: int,
                                 checked_hexes: AbstractSet[str]) -> Optional[bool]:
    hexes_to_check = {*get_hex_neighbourhood(move)} & {*list_of_moves}
    for _hex in hexes_to_check:
        if _hex not in checked_hexes:
            checked_hexes.add(_hex)
            if _hex[0] == row_number:
                return True
            else:
                if check_if_connects_to_nth_row(_hex, list_of_moves, row_number, checked_hexes):
                    return True


def check_if_connects_to_nth_column(move: Tuple[int, int], list_of_moves: Iterable[Tuple[int, int]], col_number: int,
                                    checked_hexes: AbstractSet[str]) -> Optional[bool]:
    hexes_to_check = {*get_hex_neighbourhood(move)} & {*list_of_moves}
    for _hex in hexes_to_check:
        if _hex not in checked_hexes:
            checked_hexes.add(_hex)
            if _hex[1] == col_number:
                return True
            else:
                if check_if_connects_to_nth_column(_hex, list_of_moves, col_number, checked_hexes):
                    return True


def get_winner(red_moves: Iterable[Tuple[int, int]], blue_moves: Iterable[Tuple[int, int]], last_move: Tuple[int, int],
               size: int) -> Optional[int]:
    if last_move in red_moves and check_basic_win_condition_for_red_player(red_moves, size):
        if (check_if_connects_to_nth_row(last_move, red_moves, 0, set()) and
                check_if_connects_to_nth_row(last_move, red_moves, size-1, set())):
            return 1
    elif last_move in blue_moves and check_basic_win_condition_for_blue_player(blue_moves, size):
        if (check_if_connects_to_nth_column(last_move, blue_moves, 0, set()) and
                check_if_connects_to_nth_column(last_move, blue_moves, size-1, set())):
            return -1


def negamax_alpha_beta_pruned(
    player: int,
    red_moves: Iterable[str],
    blue_moves: Iterable[str],
    last_move: str,
    alpha: float,
    beta: float,
    size: int
) -> Dict[str, int]:  # yapf: disable
    """
    Simple implementation of the negamax (minimax) algorithm for the game of hex. Includes an improvement
    of alpha-beta pruning.

    See tests for an example usage.

    :param player: the player to make a move(can be 1 or -1)
    :param red_moves: already played moves of the red player
    :param blue_moves: already played moves of the blue player
    :param last_move: last played move
    :param alpha: the minimum score that the maximizing player is assured of
    :param beta: the maximum score that the minimizing player is assured of
    :param size: size of the board
    :return: dict with results for score and move; the score is given from the perspective of the player who is about
    to play (so score == 1 when player == -1 means that player "-1" won)
    """
    winner = get_winner(red_moves, blue_moves, last_move, size)
    if winner:
        return {'score': winner * player, 'move': None}

    best_score = -np.inf

    for move in range(size**2):
        row = move // size
        col = move % size
        position = chr(col + 1 + 96) + str(1 + row)
        if position not in red_moves + blue_moves:
            copied_red_moves = red_moves.copy()
            copied_blue_moves = blue_moves.copy()
            if player == 1:
                copied_red_moves.append(position)
            else:
                copied_blue_moves.append(position)
            result = negamax_alpha_beta_pruned(-player, copied_red_moves, copied_blue_moves, position, -beta, -alpha,
                                               size)
            score = -result['score']
            if score > best_score:
                best_score = score
                best_move = (row, col)
            alpha = max(alpha, score)
            if alpha >= beta:
                break

    return {'score': best_score, 'move': best_move}


TRANSPOSITION_TABLE = {}


def negamax_alpha_beta_pruned_with_transposition_tables(
    player: int,
    red_moves: Iterable[Tuple[int, int]],
    blue_moves: Iterable[Tuple[int, int]],
    last_move: Tuple[int, int],
    alpha: float,
    beta: float,
    size: int
) -> Dict[str, int]:  # yapf: disable
    """
    Simple implementation of the negamax (minimax) algorithm for the game of hex. Includes an improvement
    of alpha-beta pruning and transposition tables.

    See tests for example usage.

    :param player: the player to make a move(can be 1 or -1)
    :param red_moves: already played moves of the red player
    :param blue_moves: already played moves of the blue player
    :param last_move: last played move
    :param alpha: the minimum score that the maximizing player is assured of
    :param beta: the maximum score that the minimizing player is assured of
    :param size: size of the board
    :return: dict with results for score and move; the score is given from the perspective of the player who is about
    to play (so score == 1 when player == -1 means that player "-1" won)
    """
    alpha_orig = alpha

    # transposition table lookup
    tt_entry = TRANSPOSITION_TABLE.get((tuple(sorted(red_moves)), tuple(sorted(blue_moves))))
    if tt_entry:
        if tt_entry['flag'] == 'EXACT':
            return {'score': tt_entry['value'], 'move': tt_entry['move']}
        elif tt_entry['flag'] == 'LOWER_BOUND':
            alpha = max(alpha, tt_entry['value'])
        elif tt_entry['flag'] == 'UPPER_BOUND':
            beta = min(beta, tt_entry['value'])
        if alpha >= beta:
            return {'score': tt_entry['value'], 'move': tt_entry['move']}

    winner = get_winner(red_moves, blue_moves, last_move, size)
    if winner:
        return {'score': winner * player, 'move': None}

    best_score = -np.inf

    for move in range(size**2):
        row = move // size
        col = move % size
        position = (row, col)
        if position not in red_moves + blue_moves:
            copied_red_moves = red_moves.copy()
            copied_blue_moves = blue_moves.copy()
            if player == 1:
                copied_red_moves.append(position)
            else:
                copied_blue_moves.append(position)
            result = negamax_alpha_beta_pruned_with_transposition_tables(
                -player, copied_red_moves, copied_blue_moves, position, -beta, -alpha, size)
            score = -result['score']
            if score > best_score:
                best_score = score
                best_move = (row, col)
            alpha = max(alpha, score)
            if alpha >= beta:
                break

    # transposition table store
    if best_score <= alpha_orig:
        flag = 'UPPER_BOUND'
    elif best_score >= beta:
        flag = 'LOWER_BOUND'
    else:
        flag = 'EXACT'
    TRANSPOSITION_TABLE[(tuple(sorted(red_moves)), tuple(sorted(blue_moves)))] = {
        'value': best_score,
        'flag': flag,
        'move': best_move
    }

    return {'score': best_score, 'move': best_move}
