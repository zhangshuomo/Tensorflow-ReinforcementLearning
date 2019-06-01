import tkinter as tk
import time

SIZE = 50
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
        h1_loc = [origin[0] + ((WIDTH - 1) // 2) * SIZE, origin[1] + ((HEIGHT + 1) // 2) * SIZE]
        h2_loc = [origin[0] + ((WIDTH + 1) // 2) * SIZE, origin[1] + ((HEIGHT - 1) // 2) * SIZE]
        self.hell1 = self.canvas.create_rectangle(h1_loc[0] - 15, h1_loc[1] - 15,
                                                  h1_loc[0] + 15, h1_loc[1] + 15,
                                                  fill='black')
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
        time.sleep(0.2)

    def reset(self):
        self.canvas.delete(self.agent)
        origin = [SIZE / 2, SIZE / 2]
        time.sleep(0.2)
        self.agent = self.canvas.create_rectangle(origin[0] - 15, origin[1] - 15,
                                     origin[0] + 15, origin[1] + 15, fill='red')
        self.render()
        return str(self.canvas.coords(self.agent))

    def step(self, action):
        pos = self.canvas.coords(self.agent)
        reward = 0
        if action == 0:
            if pos[1] + 15 > SIZE:
                self.canvas.move(self.agent, 0, -SIZE)
            else:
                reward = -0.2
        elif action == 1:
            if HEIGHT * SIZE - (pos[3] - 15) > SIZE:
                self.canvas.move(self.agent, 0, SIZE)
            else:
                reward = -0.2
        elif action == 2:
            if pos[0] + 15 > SIZE:
                self.canvas.move(self.agent, -SIZE, 0)
            else:
                reward = -0.2
        elif action == 3:
            if WIDTH * SIZE - (pos[2] - 15) > SIZE:
                self.canvas.move(self.agent, SIZE, 0)
            else:
                reward = -0.2
        new_state = self.canvas.coords(self.agent)
        if new_state in [self.canvas.coords(self.hell1), self.canvas.coords(self.hell2)]:
            reward = -1
            new_state = 'terminition'
        elif new_state == self.canvas.coords(self.treasure):
            reward = 5
            new_state = 'terminition'
        self.render()
        return str(new_state), reward
