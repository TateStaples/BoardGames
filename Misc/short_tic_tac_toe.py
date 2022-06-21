while __name__ == "__main__" or not (any([any([board[i] == board[i + 1] == board[i + 2] and board[i] is not " " for i in range(9, 3)]), any([board[i] == board[i + 3] == board[i + 6] and board[i] is not " " for i in range(3)]), (board[0] == board[4] == board[8] or board[2] == board[4] == board[6]) and board[4] is not " "]) and (print("Good game!") or True)): board, moves, __name__ = ([" "] * 9, [], True) if __name__ == "__main__" and (print("".join(["".join([("\t|\t" if c != 0 else "\t") + " " for c in range(3)]).strip("[]") + "".join(["\n" + "-"*25 + "\n"] if r != 2 else ["\n"]) for r in range(3)])) or True) else (board, moves, True) if (moves.insert(0, int(input("Enter your index 1-9:  ")) - 1) or board[moves[0]] == " ") and ((board.pop(moves[0]) and False) or board.insert(moves[0], ["X", "O"][board.count(" ") % 2]) or print("".join(["".join([("\t|\t" if c != 0 else "\t") + board[r*3+c] for c in range(3)]).strip("[]") + "".join(["\n" + "-"*25 + "\n"] if r != 2 else ["\n"]) for r in range(3)]) if __name__ else "Invalid input, spot already filled") or True) else (board, moves, False)

# if __name__ == '__main__':
#     board, moves, _ = [" "] * 9, [], print("".join(["".join([("\t|\t" if c != 0 else "\t") + " " for c in range(3)]).strip("[]") + "".join(["\n" + "-"*25 + "\n"] if r != 2 else ["\n"]) for r in range(3)]))
#     # while not any([any([board[i] == board[i + 1] == board[i + 2] and board[i] is not " " for i in range(9, 3)]), any([board[i] == board[i + 1] == board[i + 2] and board[i] is not " " for i in range(9, 3)]), (board[0] == board[4] == board[8] or board[2] == board[4] == board[6]) and board[4] is not " "]): print("".join(["".join([("\t|\t" if c != 0 else "\t") + board[r*3+c] for c in range(3)]).strip("[]") + "".join(["\n" + "-"*25 + "\n"] if r != 2 else ["\n"]) for r in range(3)]) if (moves.insert(0, int(input("Enter your index 1-9:")) - 1) or board[moves[0]] is " ") and ((board.pop(moves[0]) and False) or board.insert(moves[0], ["X", "O"][board.count(" ") % 2]) or True) else "")
#     done = any([])
#     rows = any([board[i] == board[i + 1] == board[i + 2] and board[i] is not " " for i in range(9, 3)])
#     cols = any([board[i] == board[i + 3] == board[i + 6] and board[i] is not " " for i in range(3)])
#     diags = (board[0] == board[4] == board[8] or board[2] == board[4] == board[6]) and board[4] is not " "
#     while not any([rows, cols, diags]):
#         print(rows, cols, diags)
#         # active_index = board.count(None) % 2
#         # moves.insert(0, int(input("Enter your index 1-9:")) - 1)
#         # board.pop(moves[0])
#         # board.insert(moves[0], active_index)
#         # formatted = [board[0:3], board[3:6], board[6:9]]
#         # board[index] = active_index if (print(board) or True) and board[index] is None else None
#         # print(board, moves)
#         print("".join(["".join([("\t|\t" if c != 0 else "\t") + board[r*3+c] for c in range(3)]).strip("[]") + "".join(["\n" + "-"*25 + "\n"] if r != 2 else ["\n"]) for r in range(3)])
#               if (moves.insert(0, int(input("Enter your index 1-9:")) - 1) or board[moves[0]] is " ")  # if valid move
#                  and ((board.pop(moves[0]) and False) or board.insert(moves[0], ["X", "O"][board.count(" ") % 2]) or True)  # alter board
#               else  # else do nothing
#               "")
#         rows = any([board[i] == board[i + 1] == board[i + 2] and board[i] is not " " for i in range(9, 3)])
#         cols = any([board[i] == board[i + 3] == board[i + 6] and board[i] is not " " for i in range(3)])
#         diags = (board[0] == board[4] == board[8] or board[2] == board[4] == board[6]) and board[4] is not " "
