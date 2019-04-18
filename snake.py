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
import math as m

import keras
from keras.models import load_model

np.set_printoptions(threshold=sys.maxsize)

class SnakeGame:

    def __init__(self, dim, frame_rate, square_interval):
        self.SNAKE_COLOR = (255,255,255)
        self.FOOD_COLOR = (255,100,100)   
        self.BORDER_COLOR = (75,75,75)
        self.FRAME_RATE = frame_rate

        self.x = dim
        self.y = dim
        
        self.snake_dimensions = square_interval//2
        self.i = square_interval
        self.interval = dim//square_interval

        self.model = load_model('trained models/test.h5')

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
            vision_data.append(self._get_distance((x, y), data[-1])/max_distance)

        # Food Distance
        for data in extracted_data:
            d = 0
            for p in data:
                px, py = p
                val = self.grid[px][py]
                if val == 2:
                    d = self._get_distance((x, y), p)/max_distance
                    break
            vision_data.append(d)

        # Body Part Distance
        for data in extracted_data:
            d = 0
            for p in data:
                px, py = p
                val = self.grid[px][py]
                if val == 1:
                    d = self._get_distance((x, y), p)/max_distance
                    break
            vision_data.append(d)

        # Get Food Direction
        theta = m.atan2(fy - y, fx - x)
        vision_data.append((theta/(2*m.pi)) + (1/2)) 

        return vision_data
    
    def _get_distance(self, p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        return m.sqrt(((x2-x1)**2) + ((y2-y1)**2))

    def run_game(self, games=None, is_human_player=True):

        screen = pg.display.set_mode((self.x, self.y))

        game = 0
        k = 10
        if games:
            k = games

        while game < k:
            
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
                
                    pg.display.update()
                            
                    # Check If Snake Ate Food
                    if self.curr_snake_positions[0] == self.curr_food_position:
                        fx, fy = self.curr_food_position
                        self.curr_food_position = self._get_random_location()    
                        self.snake_length += 1

                    e = pg.event.poll()
                    if e.type == pg.QUIT:
                        return

                    keys = pg.key.get_pressed()
                    
                    if keys[pg.K_ESCAPE] or keys[pg.K_SLASH]:
                        return

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

                        vision_data = self._get_vision_data()

                        x = np.asarray([self._get_training_data(self.curr_snake_positions[0])])
                        output = self.model.predict(x)
                        ai_dir = np.argmax(output[0])

                        if ai_dir == 0 and (self.curr_snake_dir != 1 or self.snake_length == 1):
                            self.curr_snake_dir = 0
                        elif ai_dir == 2 and (self.curr_snake_dir != 3 or self.snake_length == 1):
                            self.curr_snake_dir = 2
                        elif ai_dir == 1 and (self.curr_snake_dir != 0 or self.snake_length == 1):
                            self.curr_snake_dir = 1
                        elif ai_dir == 3 and (self.curr_snake_dir != 2 or self.snake_length == 1):
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
                        last_pos = self.curr_snake_positions.pop()
                        r, c = last_pos

                    # Delay to conrol frame rate
                    time.sleep(1/self.FRAME_RATE)

            print("GAME OVER.", "Score:", self.snake_length) 
            if games:
                game += 1
    

def main():
    snake = SnakeGame(500, 15, 20)
    snake.run_game(games=None, is_human_player=False)

if __name__ == "__main__":
    main()
