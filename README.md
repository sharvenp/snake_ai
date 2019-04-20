# snake_ai
A simple neural snake AI.  
\
The AI is trained using Reinforcment Learning, specifically Policy Gradients.   
\
Based on states, the action it took based on that state and the reward obtained by that action, it learns to change probablilites of actions based on the recieved reward. In other words, if an action on a state led to a low reward, it will decrease the probability of that action occuring when a similar state comes up, and vice-versa.  
\
Probability is important because the AI moves randomly based on the probability distribution of its output and hence high output values would be more likely to be chosen as the AI's move.
***
This is the AI playing after training on 28,950 games (It's pretty decent):

<p align="center">
  <img width="500" height="500" src=recording.gif>
</p>
