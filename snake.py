import keyboard 
import pygame as pg
import time
import random as r

class SnakeGame:

    def __init__(self, dim, square_interval):
        self.P1_COLOR = (255,255,255)
        self.FOOD_COLOR = (150,150,150)   
        self.BORDER_COLOR = (75,75,75)

        self.x = dim
        self.y = dim
        
        self.snake_dimensions = square_interval//2
        self.i = square_interval
        self.interval = dim//square_interval

        self._reset_game()
        
    def _get_random_location(self):
        availabe = []
        for row in range(self.interval):
            for col in range(self.interval):
                if self.grid[row][col] == 0 and (0 < row < self.interval - 1) and (0 < col < self.interval - 1):
                    availabe.append((col, row))
        return r.choice(availabe)

    def _reset_board(self):
        self.grid = []
        for row in range(self.interval):
            a = []
            for col in range(self.interval):
                a.append(0)
            self.grid.append(a)

    def _reset_game(self):
        self._reset_board()
        self.game_over = False
        self.snake_length = 1
        self.P1_POS = [(self.interval//2, self.interval//2)]
        self.food_position = self._get_random_location()    
        self.P1_DIR = r.randint(0, 3)

    def get_game_state(self):
        if self.game_over:
            return 0
        else:
            a = False
            for row in range(self.interval):
                for col in range(self.interval):
                    if self.grid[row][col] == 0:
                        a = True
            return int(a) + 1

    def run_game(self, delay, games):

        screen = pg.display.set_mode((self.x, self.y))
        
        for game in range(games):

            self._reset_game()

            while self.get_game_state():

                screen.fill((0,0,0))
                self._reset_board()

                # Draw Border
                pg.draw.rect(screen, self.BORDER_COLOR, (0, 0, self.x, self.snake_dimensions*2))
                pg.draw.rect(screen, self.BORDER_COLOR, (0, 0, self.snake_dimensions*2, self.y))
                pg.draw.rect(screen, self.BORDER_COLOR, (0, self.y - self.snake_dimensions*2, self.x, self.snake_dimensions*2))
                pg.draw.rect(screen, self.BORDER_COLOR, (self.x - self.snake_dimensions*2, 0, self.snake_dimensions*2, self.y))

                for position in self.P1_POS:
                    sx = position[0] 
                    sy = position[1] 
                    if self.grid[sy][sx] == 1 or (sx >= self.interval - 1) or (sy >= self.interval - 1) or (sx <= 0) or (sy <= 0):
                        self.game_over = True
                        break
                        
                    pg.draw.rect(screen, self.P1_COLOR, (sx*self.i, sy*self.i, self.snake_dimensions*2, self.snake_dimensions*2))                    
                    self.grid[sy][sx] = 1
                
                if self.get_game_state:
                    fx = self.food_position[0]
                    fy = self.food_position[1]            
                    pg.draw.rect(screen, self.FOOD_COLOR, (fx*self.i, fy*self.i, self.snake_dimensions*2, self.snake_dimensions*2))           
                    self.grid[fx][fy] = 2
                
                    pg.display.update()
                            
                    if self.P1_POS[0] == self.food_position:
                        self.food_position = self._get_random_location()    
                        self.snake_length += 1
                
                    if keyboard.is_pressed('w'):
                        self.P1_DIR = 0
                    elif keyboard.is_pressed('a'):
                        self.P1_DIR = 2
                    elif keyboard.is_pressed('s'):
                        self.P1_DIR = 1
                    elif keyboard.is_pressed('d'):
                        self.P1_DIR = 3

                    if keyboard.is_pressed('/'):
                        quit(0)
                    
                    pos = None
                    curr_position = self.P1_POS[0]
                    curr_x = curr_position[0]
                    curr_y = curr_position[1]
                    if self.P1_DIR == 0:
                        curr_y -= 1
                    elif self.P1_DIR == 1:
                        curr_y += 1
                    elif self.P1_DIR == 2:
                        curr_x -= 1
                    elif self.P1_DIR == 3:
                        curr_x += 1
                    pos = (curr_x % (self.interval), curr_y % (self.interval))    
                    
                    self.P1_POS.insert(0, pos)
                    if len(self.P1_POS) > self.snake_length:
                        self.P1_POS.pop()

                    time.sleep(delay)
            
            print(f"Game {game + 1} Over.", "Size:", self.snake_length)


def main():
    snake = SnakeGame(500, 20)
    snake.run_game(0.065, 3)

if __name__ == "__main__":
    main()
