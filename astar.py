import cv2
import numpy as np
import math
import heapq
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

from hillshade import get_shadow_map_stack
from info_distrib import random_info


show_animation = True

# ================================================================
# State
# ================================================================
class State:

    def __init__(self, x, y):
        self.x = x
        self.y = y

        self.state = "."   # ".","#","s","e","*"
        self.parent = None

        # A* fields
        self.g = float("inf")
        self.h = 0
        self.f = float("inf")
        
        self.t = 0

    def cost(self, other):
        """Euclidean movement cost unless obstacle."""
        if self.state == "#" or other.state == "#":
            return float("inf")
        dx = self.x - other.x
        dy = self.y - other.y
        #return math.sqrt(dx*dx + dy*dy)
        return 1

    def set_state(self, s):
        if s in ["s",".","#","e","*"]:
            self.state = s

    def __lt__(self, other):
        # Required by heapq for tie-breaking
        return self.f < other.f



# ================================================================
# Map Grid
# ================================================================
class Map:

    def __init__(self, row, col):
        self.row = row
        self.col = col
        self.map = self.init_map()

    def init_map(self):
        m = []
        for i in range(self.row):
            row = []
            for j in range(self.col):
                row.append(State(i, j))
            m.append(row)
        return m

    def get_neighbors(self, s, shadow_map):
        """8-connected neighbors."""
        res = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx = s.x + dx
                ny = s.y + dy
                
                if 0 <= nx < self.row and 0 <= ny < self.col:
                    if shadow_map[ny][nx] < 50:
                        #print('here')
                        continue
                    else:
                        res.append(self.map[nx][ny])
        return res

    def set_obstacle(self, pts):
        for x, y in pts:
            if 0 <= x < self.row and 0 <= y < self.col:
                self.map[x][y].set_state("#")



# ================================================================
# A* Planner
# ================================================================
class AStar:

    def __init__(self, grid_map):
        self.map = grid_map

    def heuristic(self, a, b):
        """Euclidean heuristic."""
        return math.sqrt((a.x - b.x)**2 + (a.y - b.y)**2)
    
    def reconstruct_time_path(self, came_from, current):
        path = []
        while current in came_from:
            x, y, t = current
            path.append((x, y))
            current = came_from[current]
        x,y,t = current
        path.append((x,y))
        px = [p[0] for p in path]
        py = [p[1] for p in path]
        px.reverse()
        py.reverse()
        return px, py

    def run(self, starts, goals, shadow_stack, num_agents):
        
        open_lists = []
        closed_sets = []
        came_from_dicts = []
        g_cost_dicts = []
        for i in range(num_agents):
            t_max = len(shadow_stack)-1
            open_list = []
            closed = set()
            came_from = {}
            g_cost = {}
            start_state = (starts[i].x, starts[i].y, 0)
            g_cost[start_state]=0
            h = self.heuristic(starts[i], goals[i])
            f = h + 0
            heapq.heappush(open_list, (f, 0, starts[i].x, starts[i].y, 0))
            open_lists.append(open_list)
            closed_sets.append(closed)
            came_from_dicts.append(came_from)
            g_cost_dicts.append(g_cost)

        finished = [False]*num_agents
        final_paths = []
        while not all(finished):
            for i in range(num_agents):
                if finished[i]==True:
                    continue
                f,g,x,y,t = heapq.heappop(open_lists[i])
                current = (x,y,t)

                if current in closed_sets[i]:
                    continue
                closed_sets[i].add(current)

                # Goal reached
                if x == goals[i].x and y==goals[i].y:
                    finished[i]=True
                    final_paths.append(self.reconstruct_time_path(came_from_dicts[i], current))
                    if all(finished):
                        break
                
                next_t = t+1
                if next_t>t_max:
                    next_t = next_t % (t_max+1)

                for neigh in self.map[i].get_neighbors(self.map[i].map[x][y], shadow_stack[next_t]):
                    nx, ny = neigh.x, neigh.y
                    if shadow_stack[next_t][ny][nx]<50:
                        continue
                    next_state = (nx, ny, next_t)
                    new_g = g+1
                    if next_state not in g_cost_dicts[i] or new_g<g_cost_dicts[i][next_state]:
                        g_cost_dicts[i][next_state] = new_g
                        came_from_dicts[i][next_state] = current
                        h = self.heuristic(neigh, goals[i])
                        f = new_g + h
                        heapq.heappush(open_lists[i], (f, new_g, nx, ny, next_t))

        return(final_paths)



# ================================================================
# Main demo
# ================================================================

def main(size, shadow_stack, start_array, end_array, num_agents):
    m_stack = []
    starts = []
    goals = []
    for n in range(num_agents):
        m = Map(size, size)

        # add boundary walls
        ox, oy = [], []
        for i in range(size):
            ox.append(i); oy.append(0)
        for i in range(size):
            ox.append(size - 1); oy.append(i)
        for i in range(size):
            ox.append(i); oy.append(size - 1)
        for i in range(size):
            ox.append(0); oy.append(i)
        m.set_obstacle(list(zip(ox, oy)))
        
        shadow_idx = np.where(shadow_stack[0]<50)
        m.set_obstacle(list(zip(list(shadow_idx[0]), list(shadow_idx[1]))))

        # start & goal
        start = m.map[start_array[n][0]][start_array[n][1]]
        goal  = m.map[end_array[n][0]][end_array[n][1]]
        m_stack.append(m)
        starts.append(start)
        goals.append(goal)

    # Run A*
    planner = AStar(m_stack)
    final_paths = planner.run(starts, goals, shadow_stack, num_agents)

    #print("Path length:", len(rx))
    #print("Path:", list(zip(rx, ry)))

    return(final_paths)

# ================================================================
# Run demo
# ================================================================


def convert_pos(pos_array, size):
    return np.round(pos_array-size/2)

def get_shadow_stack(path, time_args, bounds):
    
    time_horizon = time_args['time_horizon']
    start_time = time_args['start_time']
    end_time = time_args['end_time']
    dt = time_args['dt']

    shadow_map_stack = get_shadow_map_stack(path, 'Site01', bounds, start_time, end_time, dt)
    
    if ((end_time-start_time)//dt)!=time_horizon:
        raise Exception('Time horizon and number of time steps do not match.')
    
    shadows_idx_stack = []
    resized_shadow_stack = []
    for i in range(len(shadow_map_stack)):
        scale = 10
        shadow_map = shadow_map_stack[i]
        resized_x = shadow_map.shape[1] // scale
        resized_y = shadow_map.shape[0] // scale
        resized_shadow_map = cv2.resize(shadow_map, (resized_x, resized_y), interpolation=cv2.INTER_AREA)
        shadow_idx = np.where(resized_shadow_map<50)
        shadow_idx_array = np.array([shadow_idx[1], shadow_idx[0]]).T
        shadows_idx_stack.append(shadow_idx_array)
        resized_shadow_stack.append(resized_shadow_map)

    def padding(shadow_map, max_len, map_size):
        current_len = shadow_map.shape[0]
        padded_vals = map_size*np.ones((max_len-current_len, 2))
        return np.vstack((shadow_map, padded_vals))

    map_size =  np.shape(resized_shadow_stack[0])[0]
    max_len = max(arr.shape[0] for arr in shadows_idx_stack)
    for i in range(len(shadows_idx_stack)):
        if len(shadows_idx_stack[i])<max_len:
            shadows_idx_stack[i] =  padding(shadows_idx_stack[i], max_len, map_size)
    
    resized_shadow_stack = np.array(resized_shadow_stack)
    shadows_idx_stack = np.array(shadows_idx_stack)

    return resized_shadow_stack, shadows_idx_stack.astype(int), shadow_map_stack

def animate_plot(final_paths, start, goal, size, shadow_stack, num_agents):
    fig, ax = plt.subplots()
    shadow_mask = np.ones((size, size))
    shadow_mask[np.where(shadow_stack[0]<50)]=0
    img = ax.imshow(shadow_mask, cmap='Greys_r', animated = True)
    for i in range(num_agents):
        line, = ax.plot(final_paths[i][0][0], final_paths[i][1][0], color='r')
        ax.plot(start[i][0], start[i][1], 'ob')
        ax.plot(goal[i][0], goal[i][1], 'og')
    ax.set_xlim(0,size)
    ax.set_ylim(0,size)
    ax.set_yticks(ax.get_xticks())
    ax.set_yticklabels(ax.get_xticklabels())

    def update(frame, final_paths, line, img, num_agents, shadow_len):
        shadow_mask = np.ones((size, size))
        shadow_mask[np.where(shadow_stack[frame%shadow_len]<50)]=0
        img.set_array(shadow_mask)
        for i in range(num_agents):
            if frame>=len(final_paths[i][0]):
                continue
            line, = ax.plot(final_paths[i][0][:(frame)], final_paths[i][1][:(frame)], color='r')
        return line, img
    
    max_len = 0
    for paths in final_paths:
        if len(paths[0])>max_len:
            max_len = len(paths[0])
    shadow_len = len(shadow_stack)

    ani = FuncAnimation(fig, update, max_len, fargs=[final_paths, line, img, num_agents, shadow_len],
                                interval=200, blit=True)
    FFwriter = FFMpegWriter(fps=10, codec='libx264', bitrate=1800)
    ani.save('a*_shadow_avoidance.mp4', writer=FFwriter)



dem_path = "DEMs/Site01_final_adj_5mpp_surf.tif"
time_args = {
    'dt': 1000,
    'start_time': 0,
    'end_time': 100000,
    'time_horizon': 100
}
shadow_map_stack, shadow_idx_stack, original_shadows = get_shadow_stack(dem_path, time_args, bounds=np.array([[0, 1000], [0, 1000]]))
#size = np.shape(shadow_map_stack[0])[0]
size = np.shape(original_shadows[0])[0]
start_pos = [[50,200], [100, 300], [250, 200]]
end_pos  = [[300,280], [120, 60], [280, 50]]
num_agents = 3

final_paths = main(size, original_shadows, start_pos, end_pos, num_agents)
animate_plot(final_paths, start_pos, end_pos, size, original_shadows, num_agents)

