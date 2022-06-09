# Checkers-Engine
This engine uses the minimax algorithm with alpha-beta pruning and a slew of other optimizations to compute the best move for a position. In its current state, it is a strong player looking ahead 14-20 moves in the beginning to mid-game with 20-24 in the endgame.  
	One important note for any serious checkers players out there reading this, if you want a program to seriously improve your skills, this is likely not the one. This program does play at a pro-level, but there is a parity from the rules where kings can jump on the same turn they reach the back rank, and jumps are allowed by any piece that can jump if a double jump is possible. I like to think of this as a spin on the game! When playing other high-level engines these jumps never get used indicating that they are easily refuted but still worth mentioning.
