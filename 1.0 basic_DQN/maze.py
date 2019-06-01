import tkinter as tk
import numpy as np
import time

SIZE = 40
HEIGHT = 7
WIDTH = 7


class Maze(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('Maze')
        self.geometry("{0}x{1}".format(WIDTH * SIZE + 20, HEIGHT * SIZE + 20))
        self._construct_maze()
        # self.mainloop()

    def _construct_maze(self):
        self.canvas = tk.Canvas(self, width=WIDTH * SIZE, height=HEIGHT * SIZE, bg='white')
        for i in range(WIDTH):
            x0, y0, x1, y1 = i * SIZE, 0, i * SIZE, HEIGHT * SIZE
            self.canvas.create_line(x0, y0, x1, y1)
        for i in range(HEIGHT):
            x0, y0, x1, y1 = 0, i * SIZE, WIDTH * SIZE, i * SIZE
            self.canvas.create_line(x0, y0, x1, y1)
        origin = [SIZE / 2, SIZE / 2]
        h2_loc = [origin[0] + ((WIDTH + 1) // 2) * SIZE, origin[1] + ((HEIGHT - 1) // 2) * SIZE]
        self.hell2 = self.canvas.create_rectangle(h2_loc[0] - 15, h2_loc[1] - 15,
                                                  h2_loc[0] + 15, h2_loc[1] + 15,
                                                  fill='black')
        treasure_loc = [origin[0] + ((WIDTH + 1) // 2) * SIZE, origin[1] + ((HEIGHT + 1) // 2) * SIZE]
        self.treasure = self.canvas.create_oval(treasure_loc[0] - 15, treasure_loc[1] - 15,
                                                treasure_loc[0] + 15, treasure_loc[1] + 15,
                                                fill='yellow')
        self.agent = self.canvas.create_rectangle(origin[0] - 15, origin[1] - 15,
                                                  origin[0] + 15, origin[1] + 15,
                                                  fill='red')
        self.canvas.pack()

    def render(self):
        self.update()
        # time.sleep(0.1)

    def reset(self):
        self.canvas.delete(self.agent)
        origin = [SIZE / 2, SIZE / 2]
        time.sleep(0.1)
        self.agent = self.canvas.create_rectangle(origin[0] - 15, origin[1] - 15,
                                     origin[0] + 15, origin[1] + 15, fill='red')
        self.render()
        return np.array((self.canvas.coords(self.agent)[0]/(WIDTH*SIZE),self.canvas.coords(self.agent)[1]/(HEIGHT*SIZE)))

    def step(self, action):
        pos = self.canvas.coords(self.agent)
        reward = 0
        done = False
        if action == 0:
            if pos[1] + 15 > SIZE:
                self.canvas.move(self.agent, 0, -SIZE)
            else:
               reward = -1
        elif action == 1:
            if HEIGHT * SIZE - (pos[3] - 15) > SIZE:
                self.canvas.move(self.agent, 0, SIZE)
            else:
               reward = -1
        elif action == 2:
            if pos[0] + 15 > SIZE:
                self.canvas.move(self.agent, -SIZE, 0)
            else:
               reward = -1
        elif action == 3:
            if WIDTH * SIZE - (pos[2] - 15) > SIZE:
                self.canvas.move(self.agent, SIZE, 0)
            else:
               reward = -1
        new_state = self.canvas.coords(self.agent)
        if new_state == self.canvas.coords(self.hell2):
            reward = -3
            done = True
        elif new_state == self.canvas.coords(self.treasure):
            reward = 5
            done = True
        self.render()
        return np.array((new_state[0]/(WIDTH*SIZE),new_state[1]/(HEIGHT*SIZE))), reward, done
