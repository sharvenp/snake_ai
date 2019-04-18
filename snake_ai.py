"""
Legend:
0 - Empty Space
1 - Body Part
2 - Food
3 - Wall

real_pos - Real coordinate system position
block_pos - Grid block position

"""

import pygame as pg
import time
import random as r
import numpy as np
import sys
import os
import math as m

import keras
from keras.models import load_model
from keras.models import Model
from keras import layers
from keras import losses
from keras import optimizers
from keras import backend as keras_back
from keras import utils as np_utils


# Misc Stuff
np.set_printoptions(threshold=sys.maxsize)
save_interval = 50
food_reward = 500
death_reward = -200
idle_reward = -0.75
gamma = 0.75

class SnakeGame:

    def __init__(self, dim, square_interval):
        self.SNAKE_COLOR = (255,255,255)
        self.SNAKE_VISION_COLOR = (100,100,100)
        self.FOOD_COLOR = (255,100,100)   
        self.BORDER_COLOR = (75,75,75)

        self.x = dim
        self.y = dim
        
        self.snake_dimensions = square_interval//2
        self.i = square_interval
        self.interval = dim//square_interval

        self._create_model([25, 15, 15, 4])
        for filename in os.listdir('trained models/'):
            
        self._create_training_function()

        self._reset_game()

    def _convert_real_pos_to_block(self, pos):
        # Converts a coordinate on the screen to an interval coordinate
        x, y = pos
        return (x // self.i, y // self.i)    

    def _convert_block_pos_to_real(self, pos):
        # Converts an interval coordinate to a screen coordinate
        x, y = pos
        return (x*self.i, y*self.i)

    def _get_random_location(self):
        availabe = []

        # Get a list of all possible locations that are not occupied or on an edge
        for row in range(self.interval):
            for col in range(self.interval):
                if self.grid[row][col] == 0 and (1 < row < self.interval - 2) and (1 < col < self.interval - 2):
                    availabe.append((col, row))
        return r.choice(availabe)

    def _reset_board(self):
        
        self.grid = []
        
        # Reset grid and draw border on grid
        for row in range(self.interval):
            a = []
            for col in range(self.interval):
                if row == 0 or col == 0 or row == self.interval-1 or col == self.interval-1:
                    a.append(3)
                else:
                    a.append(0)
            self.grid.append(a)

    def _reset_game(self):

        # Reset the game and init params
        self._reset_board()
        self.game_over = False
        self.snake_length = 2
        self.curr_snake_positions = [(self.interval//2, self.interval//2)]
        self.curr_food_position = self._get_random_location()
        self.curr_snake_dir = r.randint(0, 3)

    def get_game_state(self):
    
        """
        0 - Game is over
        1 - Game is in progress
        2 - Player won
        """
    
        if self.game_over:
            return 0
        else:
            a = False
            for row in range(self.interval):
                for col in range(self.interval):
                    if self.grid[row][col] == 0:
                        a = True
            return int(a) + 1

    def _check_legal(self, pos):
        x, y = pos
        mx, my = self._convert_real_pos_to_block((self.x, self.y))
        return (x >= 0 and y >= 0 and x <= mx - 1 and y <= mx - 1)

    def _get_squares_in_direction(self, pos, direction):
        curr_pos = (pos[0] + direction[0], pos[1] + direction[1])
        a = []
        while self._check_legal(curr_pos):
            a.append(curr_pos)
            curr_pos = (curr_pos[0] + direction[0], curr_pos[1] + direction[1])
        return a

    def _get_vision_data(self):

        x, y = self.curr_snake_positions[0]
        fx, fy = self.curr_food_position
        
        max_distance = self._get_distance((1,1), self._convert_real_pos_to_block((self.x, self.y)))

        vision_data = []
        extracted_data = []       

        # Get squares by traversing in all 8 directions
        extracted_data.append(self._get_squares_in_direction((x,y), ( 0, -1))) # Left
        extracted_data.append(self._get_squares_in_direction((x,y), (-1, -1))) # TL
        extracted_data.append(self._get_squares_in_direction((x,y), (-1,  0))) # Up
        extracted_data.append(self._get_squares_in_direction((x,y), (-1,  1))) # TR
        extracted_data.append(self._get_squares_in_direction((x,y), ( 0,  1))) # Right
        extracted_data.append(self._get_squares_in_direction((x,y), ( 1,  1))) # BR
        extracted_data.append(self._get_squares_in_direction((x,y), ( 1,  0))) # Down
        extracted_data.append(self._get_squares_in_direction((x,y), ( 1, -1))) # BL

        # Wall Distance
        for data in extracted_data:
            if data:
                vision_data.append(self._get_distance((x, y), data[-1])/max_distance)
            else:
                vision_data.append(0)

        # Food Distance
        for data in extracted_data:
            if data:
                d = 0
                for p in data:
                    px, py = p
                    val = self.grid[px][py]
                    if val == 2:
                        d = self._get_distance((x, y), p)/max_distance
                        break
                vision_data.append(d)
            else:
                vision_data.append(0)

        # Body Part Distance
        for data in extracted_data:
            if data:
                d = 0
                for p in data:
                    px, py = p
                    val = self.grid[px][py]
                    if val == 1:
                        d = self._get_distance((x, y), p)/max_distance
                        break
                vision_data.append(d)
            else:
                vision_data.append(0)

        # Get Food Direction
        theta = m.atan2(fy - y, fx - x)
        vision_data.append((theta/(2*m.pi)) + (1/2)) 

        return np.asarray(vision_data), extracted_data
    
    def _get_distance(self, p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        return m.sqrt(((x2-x1)**2) + ((y2-y1)**2))

    def run_game(self, frame_rate, games, is_human_player=True, draw_snake_vision=False):

        screen = pg.display.set_mode((self.x, self.y))

        game = 0

        states = None
        actions = None
        rewards = None

        if not is_human_player:
            states, actions, rewards = [], [], []

        learning_interval = 100

        while game < games:
            
            self._reset_game()

            while self.get_game_state():

                screen.fill((0,0,0))
                self._reset_board()

                # Draw Border
                pg.draw.rect(screen, self.BORDER_COLOR, (0, 0, self.x, self.snake_dimensions*2))
                pg.draw.rect(screen, self.BORDER_COLOR, (0, 0, self.snake_dimensions*2, self.y))
                pg.draw.rect(screen, self.BORDER_COLOR, (0, self.y - self.snake_dimensions*2, self.x, self.snake_dimensions*2))
                pg.draw.rect(screen, self.BORDER_COLOR, (self.x - self.snake_dimensions*2, 0, self.snake_dimensions*2, self.y))

                for position in self.curr_snake_positions:
                    sx, sy = position

                    # Check If Snake Ate Itself
                    if self.grid[sy][sx] == 1 or sx <= 0 or sy <= 0 or sx >= self.interval-1 or sy >= self.interval-1:
                        self.game_over = True
                        if not is_human_player:
                            rewards.append(death_reward)
                        break
                        
                    # Draw Snake
                    rx, ry = self._convert_block_pos_to_real(position)
                    pg.draw.rect(screen, self.SNAKE_COLOR, (rx, ry, self.snake_dimensions*2, self.snake_dimensions*2))                    
                    self.grid[sy][sx] = 1

                if self.get_game_state:

                    # Draw Food
                    fx, fy = self.curr_food_position
                    rfx, rfy = self._convert_block_pos_to_real(self.curr_food_position)
                    pg.draw.rect(screen, self.FOOD_COLOR, (rfx, rfy, self.snake_dimensions*2, self.snake_dimensions*2))           
                    self.grid[fy][fx] = 2
                                            
                    # Check If Snake Ate Food
                    if self.curr_snake_positions[0] == self.curr_food_position:
                        fx, fy = self.curr_food_position
                        self.curr_food_position = self._get_random_location()    
                        self.snake_length += 1
                        
                        if not is_human_player:
                            rewards.append(food_reward)
                    else:
                        rewards.append(idle_reward)

                    e = pg.event.poll()
                    if e.type == pg.QUIT:
                        return

                    keys = pg.key.get_pressed()
                    
                    if keys[pg.K_ESCAPE] or keys[pg.K_SLASH]:
                        return

                    if not is_human_player and len(states) % learning_interval == 0 and states:
                        
                        self._train_episode(states, actions, rewards, game)
                        states, actions, rewards = [], [], []


                    if is_human_player: # Use Keyboard if human is playing

                        # Get Keyboard Input
                        if keys[pg.K_w] and self.curr_snake_dir != 1:
                            self.curr_snake_dir = 0
                        elif keys[pg.K_a] and self.curr_snake_dir != 3:
                            self.curr_snake_dir = 2
                        elif keys[pg.K_s] and self.curr_snake_dir != 0:
                            self.curr_snake_dir = 1
                        elif keys[pg.K_d] and self.curr_snake_dir != 2:
                            self.curr_snake_dir = 3

                    else: # Use neural net

                        vision_data, extracted_data = self._get_vision_data()
                        
                        if draw_snake_vision:
                            for data in extracted_data:
                                if data:
                                    px, py = self._convert_block_pos_to_real(data[-1])
                                    cx, cy = self._convert_block_pos_to_real(self.curr_snake_positions[0])
                                    pg.draw.line(screen, self.SNAKE_VISION_COLOR, 
                                                (cx+self.snake_dimensions, cy+self.snake_dimensions),
                                                (px+self.snake_dimensions, py+self.snake_dimensions))   
                        states.append(vision_data)
                        action = self._get_state_action(vision_data)
                        actions.append(action)

                        if action == 0 and (self.curr_snake_dir != 1 or self.snake_length == 1):
                            self.curr_snake_dir = 0
                        elif action == 2 and (self.curr_snake_dir != 3 or self.snake_length == 1):
                            self.curr_snake_dir = 2
                        elif action == 1 and (self.curr_snake_dir != 0 or self.snake_length == 1):
                            self.curr_snake_dir = 1
                        elif action == 3 and (self.curr_snake_dir != 2 or self.snake_length == 1):
                            self.curr_snake_dir = 3

                    # Update Direction Vector Based on Movement State
                    if self.curr_snake_dir == 0:
                        self.direction_vector = (0, -1)
                    elif self.curr_snake_dir == 1:
                        self.direction_vector = (0, 1)
                    elif self.curr_snake_dir == 2:
                        self.direction_vector = (-1, 0)
                    elif self.curr_snake_dir == 3:
                        self.direction_vector = (1, 0)
                    
                    dx, dy = self.direction_vector
                    curr_x, curr_y = self.curr_snake_positions[0]
                    pos = ((curr_x + dx) % self.interval, (curr_y + dy) % (self.interval))    
                    
                    # Pop end of snake to maintain size
                    self.curr_snake_positions.insert(0, pos)
                    if len(self.curr_snake_positions) > self.snake_length:
                        self.curr_snake_positions.pop()

                    pg.display.update()

                    # Delay to conrol frame rate
                    time.sleep(1/frame_rate)


            if len(rewards) != len(states) and rewards:
                rewards.pop()    
            print(f"EPISODE {game} OVER.", "Score:", self.snake_length - 2, "TOTAL REWARD:", sum(rewards))
            game += 1

            if not is_human_player and states:
                self._train_episode(states, actions, rewards, game)
                states, actions, rewards = [], [], []

    
    
    # ---------- AI Methods ---------- #
    def _create_model(self, architecture):

        self.input_dim = architecture[0]
        self.output_dim = architecture[-1]
        self.input = layers.Input(shape=(self.input_dim,))
        net = self.input
        
        for h_dim in architecture[1:len(architecture)-1]:
            net = layers.Dense(h_dim)(net)
            net = layers.Activation("relu")(net)

        net = layers.Dense(self.output_dim)(net)
        net = layers.Activation("softmax")(net)
        self.model = Model(inputs=self.input, outputs=net)

    def _load_model(self, directory):
        self.model = load_model(directory)
        print("Loaded Model From:", directory)


    def _save_model(self, directory):
        self.model.save(directory)
        print("Created Model. Saved to:", directory)


    def _create_training_function(self):
        
        action_prob_placeholder = self.model.output
        action_onehot_placeholder = keras_back.placeholder(shape=(None, self.output_dim), name="action_onehot")
        discount_reward_placeholder = keras_back.placeholder(shape=(None,), name="discount_reward")

        action_prob = keras_back.sum(action_prob_placeholder * action_onehot_placeholder, axis=1)
        log_action_prob = keras_back.log(action_prob)

        loss = -1*log_action_prob * discount_reward_placeholder
        loss = keras_back.mean(loss)

        adam = optimizers.Adam()

        updates = adam.get_updates(params=self.model.trainable_weights,
                                   loss=loss)

        self.training_function = keras_back.function(inputs=[self.model.input, action_onehot_placeholder,
                                                     discount_reward_placeholder], outputs=[],
                                                     updates=updates)

    def _get_state_action(self, state_input):
        action_prob = np.squeeze(self.model.predict(np.asarray([state_input])))
        val = np.random.choice(np.arange(self.output_dim), p=action_prob)
        # print(action_prob)
        # val = np.argmax(action_prob)
        return val
    
    def _train (self, S, A, R):
        """Train a network
        Args:
            S (2-D Array): `state` array of shape (n_samples, state_dimension)
            A (1-D Array): `action` array of shape (n_samples,)
                It's simply a list of int that stores which actions the agent chose
            R (1-D Array): `reward` array of shape (n_samples,)
                A reward is given after each action.
        """
        action_onehot = np_utils.to_categorical(A, num_classes=self.output_dim)
        discount_reward = self._compute_discounted_rewards(R, discount_rate=gamma)
        self.training_function([S, action_onehot, discount_reward])

    def _train_episode(self, s, a, r, game):
        while len(r) != len(s) and r:
            r.pop()    
        directory = 'trained models/'
        states = np.array(s)
        actions = np.array(a)
        rewards = np.array(r)
        self._train(states, actions, rewards)
        if game % save_interval == 0 and game:
            self._save_model(directory+"episode-"+str(game)+"-snake.h5")

    def _compute_discounted_rewards (self, R, discount_rate=.99):
        """Returns discounted rewards
        Args:
            R (1-D array): a list of `reward` at each time step
            discount_rate (float): Will discount the future value by this rate
        Returns:
            discounted_r (1-D array): same shape as input `R`
                but the values are discounted
        """
        discounted_r = np.zeros_like(R, dtype=np.float32)
        running_add = 0
        for t in reversed(range(len(R))):
            running_add = running_add * discount_rate + R[t]
            discounted_r[t] = running_add
        discounted_r -= (discounted_r.mean() / discounted_r.std())
        return discounted_r

def main():
    snake = SnakeGame(500, 20)
    snake.run_game(90, 1000000, is_human_player=False, draw_snake_vision=False)

if __name__ == "__main__":
    main()
