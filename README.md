# Checkers-Engine
This engine uses the minimax algorithm with alpha-beta pruning and a slew of other optimizations to compute the best move for a position. In its current state, it is a
strong player looking ahead quite far with the extended search reaching 30 moves deep. In fact, it is able to tie cake sometimes (cake is a very strong engine for
reference) when it is given time odds of 2:1
and cake is not allowed an end-game table (so only when cake is crippled but hey I'm still happy about this). 

In its current state, the program needs some love, and many mistakes that I made as I started this project still lurk in the depths of the code. There is one
inconsistency with the rules of checkers that should be noted, kings can jump immediately after getting to the last rank this is quite a rare
occurrence though at high-level engine play and seems to happen more and more the worse the opponent! I challenge anyone to beat it and if you do I would love to hear
your opinions on its play.
