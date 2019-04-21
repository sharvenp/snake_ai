# snake_ai
A simple neural snake AI.  
\
The AI is trained using Reinforcement Learning, specifically Policy Gradients.   
\
Based on states, the action it took based on that state and the reward obtained by that action, it learns to change probablilites of actions based on the recieved reward. In other words, if an action on a state led to a low reward, it will decrease the probability of that action occuring when a similar state comes up, and vice-versa.  
\
Probability is important because the AI moves randomly based on the probability distribution of its output and hence high output values would be more likely to be chosen as the AI's move.
***
This is the AI playing after training on 28,950 games (It's pretty decent):

<p align="center">
  <img width="500" height="500" src=recording.gif>
</p>

# Dependencies
- keras
- pygame
- termcolor (print colored text to terminal 😎)

# Usage
- Run snake_ai.py and make sure there is a "trained models" folder in the same directory as the script; this is where the trained models will be saved. After running, a window will appear and the agent will continuously play and learn. The parameters of the game can be adjusted in the __init__ of the main class in the script
//
- Run keyboard_snake.py if you just want to play snake 😄 (controls are WASD or Arrow Keys)
