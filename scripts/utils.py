import heapq
import random

from typing import AbstractSet, Dict, Iterable, List, Optional, Tuple

import numpy as np


TRANSPOSITION_TABLE = {}


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


def get_valid_hex_neighbourhood(position: Tuple[int, int], size: int) -> List[Tuple[int, int]]:
    # TODO write tests
    """
    Returns (only valid) coordinates of adjacent hexes.
    """
    return [hex_ for hex_ in get_hex_neighbourhood(position) if 0 <= hex_[0] <= size-1 and 0 <= hex_[1] <= size-1]


def check_basic_win_condition_for_red_player(list_of_moves: Iterable[Tuple[int, int]], size: int) -> bool:
    rows = set([move[0] for move in list_of_moves])
    return len(rows) == size


def check_basic_win_condition_for_blue_player(list_of_moves: Iterable[Tuple[int, int]], size: int) -> bool:
    cols = set([move[1] for move in list_of_moves])
    return len(cols) == size


def check_if_connects_to_nth_row(move: Tuple[int, int], list_of_moves: Iterable[Tuple[int, int]], row_number: int,
                                 checked_hexes: AbstractSet[str]) -> Optional[bool]:
    hexes_to_check = {*get_hex_neighbourhood(move)} & {*list_of_moves}
    for hex_ in hexes_to_check:
        if hex_ not in checked_hexes:
            checked_hexes.add(hex_)
            if hex_[0] == row_number:
                return True
            else:
                if check_if_connects_to_nth_row(hex_, list_of_moves, row_number, checked_hexes):
                    return True


def check_if_connects_to_nth_column(move: Tuple[int, int], list_of_moves: Iterable[Tuple[int, int]], col_number: int,
                                    checked_hexes: AbstractSet[str]) -> Optional[bool]:
    hexes_to_check = {*get_hex_neighbourhood(move)} & {*list_of_moves}
    for hex_ in hexes_to_check:
        if hex_ not in checked_hexes:
            checked_hexes.add(hex_)
            if hex_[1] == col_number:
                return True
            else:
                if check_if_connects_to_nth_column(hex_, list_of_moves, col_number, checked_hexes):
                    return True


def get_winner(red_moves: Iterable[Tuple[int, int]], blue_moves: Iterable[Tuple[int, int]], last_move: Tuple[int, int],
               size: int) -> Optional[int]:
    if last_move in red_moves and check_basic_win_condition_for_red_player(red_moves, size):
        if (check_if_connects_to_nth_row(last_move, red_moves, 0, set()) and
                check_if_connects_to_nth_row(last_move, red_moves, size-1, set())):
            return 100
    elif last_move in blue_moves and check_basic_win_condition_for_blue_player(blue_moves, size):
        if (check_if_connects_to_nth_column(last_move, blue_moves, 0, set()) and
                check_if_connects_to_nth_column(last_move, blue_moves, size-1, set())):
            return -100


def get_available_positions(occupied_positions: Iterable[Tuple[int, int]], size: int) -> List[Tuple[int, int]]:
    # TODO write tests
    """
    Returns available (legal) positions on the board.

    Assumption: the board is symmetrical in size (so e.g. 11x11, not 10x11).
    """
    available_positions = []
    for move in range(size ** 2):
        position = (move // size, move % size)
        if position not in occupied_positions:
            available_positions.append(position)
    return available_positions


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

    See tests for example usage.

    :param player: the player to make a move (can be 1 or -1)
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

    available_positions = get_available_positions(red_moves+blue_moves, size)
    for position in available_positions:
        copied_red_moves = red_moves.copy()
        copied_blue_moves = blue_moves.copy()
        if player == 1:
            copied_red_moves.append(position)
        else:
            copied_blue_moves.append(position)
        result = negamax_alpha_beta_pruned(-player, copied_red_moves, copied_blue_moves, position, -beta, -alpha, size)
        score = -result['score']
        if score > best_score:
            best_score = score
            best_move = position
        alpha = max(alpha, score)
        if alpha >= beta:
            break

    return {'score': best_score, 'move': best_move}


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

    :param player: the player to make a move (can be 1 or -1)
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

    available_positions = get_available_positions(red_moves+blue_moves, size)
    for position in available_positions:
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
            best_move = position
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


class PriorityQueue:
    def __init__(self):
        self.elements = []
        # heapq.heapify(self.elements) ?

    def empty(self):
        return len(self.elements) == 0

    def put(self, item, priority):
        heapq.heappush(self.elements, (priority, item))

    def get(self):
        return heapq.heappop(self.elements)[1]


def reconstruct_path(came_from, goal, occupied_edges, player, graph):
    current = goal
    path = []
    if player == 1:
        starting_edges = graph.red_bottom_edges
    else:
        starting_edges = graph.blue_right_edges
    while current not in starting_edges:
        if current not in occupied_edges:
            path.append(current)
        current = came_from[current]
    if current not in occupied_edges:
        path.append(current)
    path.reverse()
    return path


class BoardGraph:
    def __init__(self, size):
        self.size = size
        self.edges = {}
        self.red_moves = []
        self.blue_moves = []
        self.red_bottom_edges = []
        self.blue_right_edges = []

    def populate_edges_and_neighbours(self):
        for hex_ in range(self.size ** 2):
            position = (hex_ // self.size, hex_ % self.size)
            self.edges[position] = get_valid_hex_neighbourhood(position, self.size)
        self.red_bottom_edges = [edge for edge in self.edges if edge[0] == self.size - 1]
        self.blue_right_edges = [edge for edge in self.edges if edge[1] == self.size - 1]

    def append_move(self, red_move_to_append=None, blue_move_to_append=None):
        if red_move_to_append:
            self.red_moves.append(red_move_to_append)
            try:
                self.blue_right_edges.remove(red_move_to_append)
            except ValueError:
                # the move does not belong to edge hexes
                pass
        elif blue_move_to_append:
            self.blue_moves.append(blue_move_to_append)
            try:
                self.red_bottom_edges.remove(blue_move_to_append)
            except ValueError:
                # the move does not belong to edge hexes
                pass

    def valid_adjacent_edges(self, edge, opponent_moves):
        return [edge for edge in self.edges[edge] if edge not in opponent_moves]

    @staticmethod
    def get_movement_cost(target_node, player_moves):
        # nodes are either unoccupied or of the player's color
        if target_node in player_moves:
            cost = 0
        else:
            cost = 1
        return cost

    def get_winner(self):
        if len(self.red_moves) > len(self.blue_moves):  # red to play; doesn't take into account swap!!!
            if check_basic_win_condition_for_red_player(self.red_moves, self.size):
                if (check_if_connects_to_nth_row(self.red_moves[-1], self.red_moves, 0, set()) and
                        check_if_connects_to_nth_row(self.red_moves[-1], self.red_moves, self.size-1, set())):
                    return 100
        else:
            if check_basic_win_condition_for_blue_player(self.blue_moves, self.size):
                if (check_if_connects_to_nth_column(self.blue_moves[-1], self.blue_moves, 0, set()) and
                        check_if_connects_to_nth_column(self.blue_moves[-1], self.blue_moves, self.size-1, set())):
                    return -100

    def get_available_positions(self):
        return [edge for edge in self.edges.keys() if edge not in self.red_moves and edge not in self.blue_moves]

    def copy(self):
        new_object = BoardGraph(self.size)
        new_object.edges = self.edges.copy()
        new_object.red_moves = self.red_moves.copy()
        new_object.blue_moves = self.blue_moves.copy()
        new_object.red_bottom_edges = self.red_bottom_edges.copy()
        new_object.blue_right_edges = self.blue_right_edges.copy()
        return new_object


def dijkstra_search(graph, player):
    frontier = PriorityQueue()
    came_from = {}
    cost_so_far = {}

    if player == 1:
        starting_edges = graph.red_bottom_edges
        player_moves = graph.red_moves
        opponent_moves = graph.blue_moves
    else:
        starting_edges = graph.blue_right_edges
        player_moves = graph.blue_moves
        opponent_moves = graph.red_moves

    for start in starting_edges:
        if start in player_moves:
            frontier.put(start, 0)
            cost_so_far[start] = 0
        else:
            frontier.put(start, 1)
            cost_so_far[start] = 1
        came_from[start] = None

    while not frontier.empty():
        current = frontier.get()

        if player == 1 and current[0] == 0 or player == -1 and current[1] == 0:
            break

        for next_ in graph.valid_adjacent_edges(current, opponent_moves):
            new_cost = cost_so_far[current] + graph.get_movement_cost(next_, player_moves)
            if new_cost < cost_so_far.get(next_, np.inf):
                cost_so_far[next_] = new_cost
                priority = new_cost
                frontier.put(next_, priority)
                came_from[next_] = current

    return came_from, cost_so_far


def run_dijkstra_and_get_cost_to_goal(board_graph, player):
    came_from, cost_so_far = dijkstra_search(board_graph, player)
    if player == 1:
        goal_edges = [edge for edge in cost_so_far if edge[0] == 0]
    else:
        goal_edges = [edge for edge in cost_so_far if edge[1] == 0]
    goal_with_the_lowest_cost = min(goal_edges, key=lambda x: cost_so_far[x])
    lowest_cost = cost_so_far[goal_with_the_lowest_cost]
    # shortest_path = reconstruct_path(came_from, goal_with_the_lowest_cost, red_moves+blue_moves, player, board_graph)
    return lowest_cost


def get_heuristic_score(board_graph, player):
    # positive value indicate player is winning
    # negative values indicate player is losing
    # 0 means that both players need equal number of moves to win - but the `player` is yet about to move
    # so basically he wins in this scenario as well - that's why we add there 0.5
    return (run_dijkstra_and_get_cost_to_goal(board_graph, -player) -
            run_dijkstra_and_get_cost_to_goal(board_graph, player) + 0.5)


def get_center_fields(size):
    min_= size // 2
    max_ = min_ + 1
    return [(min_, min_), (min_, max_), (max_, min_), (max_, max_)]


def negamax_alpha_beta_pruned_with_transposition_tables_and_heuristics(
    player: int,
    board: BoardGraph,
    alpha: float,
    beta: float,
    depth: int
) -> Dict[str, int]:  # yapf: disable
    """
    Simple implementation of the negamax (minimax) algorithm for the game of hex. Includes an improvement
    of alpha-beta pruning and transposition tables.

    See tests for example usage.

    :param player: the player to make a move (can be 1 or -1)
    :param board: object with the graph representation of the board; contains already played moves
    :param alpha: the minimum score that the maximizing player is assured of
    :param beta: the maximum score that the minimizing player is assured of
    :param depth: ...
    :return: dict with results for score and move; the score is given from the perspective of the player who is about
    to play (so score == 1 when player == -1 means that player "-1" won)
    """
    alpha_orig = alpha

    # transposition table lookup
    tt_entry = TRANSPOSITION_TABLE.get((tuple(sorted(board.red_moves)), tuple(sorted(board.blue_moves))))
    if tt_entry and tt_entry['depth'] >= depth:
        if tt_entry['flag'] == 'EXACT':
            return {'score': tt_entry['value'], 'move': tt_entry['move']}
        elif tt_entry['flag'] == 'LOWER_BOUND':
            alpha = max(alpha, tt_entry['value'])
        elif tt_entry['flag'] == 'UPPER_BOUND':
            beta = min(beta, tt_entry['value'])
        if alpha >= beta:
            return {'score': tt_entry['value'], 'move': tt_entry['move']}

    winner = board.get_winner()
    if winner:
        return {'score': winner * player, 'move': None}

    if depth == 0:
        return {'score': get_heuristic_score(board, player), 'move': None}

    best_score = -np.inf

    available_positions = board.get_available_positions()
    for position in available_positions:
        new_board = board.copy()
        if player == 1:
            new_board.append_move(red_move_to_append=position)
        else:
            new_board.append_move(blue_move_to_append=position)
        result = negamax_alpha_beta_pruned_with_transposition_tables_and_heuristics(
            -player, new_board, -beta, -alpha, depth-1)
        score = -result['score']
        if score > best_score:
            best_score = score
            best_move = position
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
    TRANSPOSITION_TABLE[(tuple(sorted(board.red_moves)), tuple(sorted(board.blue_moves)))] = {
        'value': best_score,
        'flag': flag,
        'move': best_move,
        'depth': depth
    }

    return {'score': best_score, 'move': best_move}
