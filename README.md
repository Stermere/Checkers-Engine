# Checkers-Engine
A checkers engine that using the MinMax algorithm along with Monty Carlo tree search to find a optimal move for the current position.

To play against it, install python 3.7 or above and pip install pygame 2.0.0 or above and run Main.py, then have fun!

There are two variables you might want to change 1. P_TIME will allow the bot to think for more or less time (default is 2.5 seconds)
 2. BOT_PLAYING will allow you to have the bot playing against itself if set to True. Both variables can be found at line 308 in main.

Testing has shown that 0.5 seocnds is hard but beatable, 2.5 seconds is very hard and 10 is overkill
