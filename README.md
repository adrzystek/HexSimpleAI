This project aims at creating a decent automated player (let's call it an "AI") at the game 
of [hex](https://en.wikipedia.org/wiki/Hex_(board_game)). Hopefully, one day it will be 
implemented at my site [hexy.games](https://hexy.games/).

For now it is the [negamax](https://en.wikipedia.org/wiki/Negamax) algorithm with 
the improvements of [alpa-beta pruning](https://en.wikipedia.org/wiki/Alpha%E2%80%93beta_pruning)
and [transposition tables](https://en.wikipedia.org/wiki/Transposition_table). Heuristics are 
on the way!

Plays perfectly on board size 4x4 (thinking time up to 50 seconds on an ordinary notebook).

Take a look at [tests](https://github.com/adrzystek/HexSimpleAI/blob/master/tests/test_utils.py) 
for examples of use.
