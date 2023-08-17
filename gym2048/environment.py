import sys

import numpy as np


def add_tile(board, rng):
    """
    Add a tile to the given board and return it.

    :param board: The board represented as a numpy array.
    :param rng: The random number generator.
    :return: The updated board.
    """
    board = np.copy(board)
    empty_squares = np.argwhere((board == 0))
    if len(empty_squares) == 0:
        return board
    idx = empty_squares[rng.integers(0, len(empty_squares))]
    num = rng.choice(np.array([2, 4]), p=np.array([0.9, 0.1]))
    board[idx[0], idx[1]] = num
    return board


def initialize(rng):
    """
    Create a new board.

    :param rng: The random number generator.
    :return: A new board array.
    """
    board = np.zeros((4, 4))
    board = add_tile(board, rng)
    board = add_tile(board, rng)
    return board


def up(board, rng):
    original_board = np.copy(board)
    board = np.copy(board)
    s = board.shape[0]
    for j in range(s):
        section = board.T[j]
        for i in range(s - 1):
            if section[i] == 0:
                section[i:-1] = section[i + 1 :]
                section[-1] = 0
        for i in range(s - 1):
            if section[i] == section[i + 1] and section[i] != 0:
                section[i] = section[i] + section[i + 1]
                section[i + 1] = 0
        for i in range(s - 1):
            if section[i] == 0:
                section[i:-1] = section[i + 1 :]
                section[-1] = 0
        board[:, j] = section
    if np.array_equal(original_board, board):
        return original_board, False
    board = add_tile(board, rng)
    return board, True


def down(board, rng):
    original_board = np.copy(board)
    board = np.copy(board)
    board = np.flipud(board)
    s = board.shape[0]
    for j in range(s):
        section = board.T[j]
        for i in range(s - 1):
            if section[i] == 0:
                section[i:-1] = section[i + 1 :]
                section[-1] = 0
        for i in range(s - 1):
            if section[i] == section[i + 1] and section[i] != 0:
                section[i] = section[i] + section[i + 1]
                section[i + 1] = 0
        for i in range(s - 1):
            if section[i] == 0:
                section[i:-1] = section[i + 1 :]
                section[-1] = 0
        board[:, j] = section
    board = np.flipud(board)
    if np.array_equal(original_board, board):
        return original_board, False
    board = add_tile(board, rng)
    return board, True


def left(board, rng):
    original_board = np.copy(board)
    board = np.copy(board)
    s = board.shape[0]
    for j in range(s):
        section = board[j]
        for i in range(s - 1):
            if section[i] == 0:
                section[i:-1] = section[i + 1 :]
                section[-1] = 0
        for i in range(s - 1):
            if section[i] == section[i + 1] and section[i] != 0:
                section[i] = section[i] + section[i + 1]
                section[i + 1] = 0
        for i in range(s - 1):
            if section[i] == 0:
                section[i:-1] = section[i + 1 :]
                section[-1] = 0
        board[j, :] = section
    if np.array_equal(original_board, board):
        return original_board, False
    board = add_tile(board, rng)
    return board, True


def right(board, rng):
    original_board = np.copy(board)
    board = np.copy(board)
    board = np.fliplr(board)
    s = board.shape[0]
    for j in range(s):
        section = board[j]
        for i in range(s - 1):
            if section[i] == 0:
                section[i:-1] = section[i + 1 :]
                section[-1] = 0
        for i in range(s - 1):
            if section[i] == section[i + 1] and section[i] != 0:
                section[i] = section[i] + section[i + 1]
                section[i + 1] = 0
        for i in range(s - 1):
            if section[i] == 0:
                section[i:-1] = section[i + 1 :]
                section[-1] = 0
        board[j, :] = section
    board = np.fliplr(board)
    if np.array_equal(original_board, board):
        return original_board, False
    board = add_tile(board, rng)
    return board, True


def game_won(board):
    return bool(np.isin(2048, board))


def game_lost(board):
    rng = np.random.default_rng(0)
    return not game_won(board) and not (
        up(board, rng)[1] == True
        or down(board, rng)[1] == True
        or left(board, rng)[1] == True
        or right(board, rng)[1] == True
    )


def move(board, action, rng):
    """
    Perform an action on the given board.

    * board: The board to make the move on.
    * action: The action to take 0: up, 1: down, 2: left, 3: right
    * rng: The random generator

    * return: The new board.
    """
    if action == 0:
        board, _ = up(board, rng)
    if action == 1:
        board, _ = down(board, rng)
    if action == 2:
        board, _ = left(board, rng)
    if action == 3:
        board, _ = right(board, rng)
    return board


def status(board):
    if game_won(board):
        return 1
    elif game_lost(board):
        return -1
    else:
        return 0


def score(board):
    return np.max(board)


def possible_moves(board):
    rng = np.random.default_rng()
    moves = []
    boards = []
    up_board, up_bool = up(board, rng)
    down_board, down_bool = down(board, rng)
    left_board, left_bool = left(board, rng)
    right_board, right_bool = right(board, rng)
    if up_bool:
        moves.append(0)
        boards.append(up_board)
    if down_bool:
        moves.append(1)
        boards.append(down_board)
    if left_bool:
        moves.append(2)
        boards.append(left_board)
    if right_bool:
        moves.append(3)
        boards.append(right_board)
    return np.array(moves), boards


def to_onehot(board):
    log_board = np.log2(board, where=(board > 0))
    unique_values = int(np.log2(2048)) + 1
    one_hot = np.zeros((*board.shape, unique_values))
    try:
        for x, y in np.argwhere(board > 0):
            one_hot[x, y, int(np.log2(board[x, y]))] = 1
    except IndexError:
        print(board)
        sys.exit()
    return one_hot


def from_onehot(one_hot):
    tmp = np.argmax(one_hot, axis=-1)
    tmp = 2**tmp
    tmp[tmp == 1] = 0
    print(tmp)
    return tmp
