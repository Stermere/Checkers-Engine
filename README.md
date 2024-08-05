# Checkers-Engine (Marcher Engine)
This engine uses the neg-max algorithm with alpha-beta pruning and several other optimizations to compute the best move for a position. In its current state, it is a solid player. The best and easiest way to play against it is to visit its web app at [my website](https://ckees1.pythonanywhere.com) the source code for which is also available on my GitHub!

## Web Application

For an optimal experience, play against the Marcher Engine through the web application:

- **Web App**: [Play Here](https://ckees1.pythonanywhere.com)

The Web App's max playing strength is strong but not unbeatable by any measure, for better play run the engine locally with a larger time control.

## How to set up to play against 
1. **Clone the repository:**
   - Run the following command to clone the repository from GitHub:
     ```
     git clone https://github.com/Stermere/Checkers-Engine
     ```

2. **Run `initVenv.bat`:**
   - Navigate to the directory where you cloned the repository in the previous step.
   - Run the `initVenv.bat` script. This will set up a Virtual environment and install the requirements.

3. **Run `build.bat`:** (Skip this step if not modifying the C code)
   - After initializing the virtual environment, run the `build.bat` script. This will Package the C code into a Python module

4. **Run `run.bat`:**
   - Finally, to play against Marcher Engine on your local machine, execute the `run.bat` script. This script will host the website on your local network. 
