# inspired from https://github.com/Mandaranian-Cactus/Scribble-Algorithm/blob/main/Poisson%20Disk%20Sampling.js
import math
from collections import deque
import numpy as np

# Create an even yet random distribution of points
def uniform_poisson_disk_sampling(W,H, min_dist=16, max_tries=30):
    cell_sz = min_dist / math.sqrt(2)
    queue = deque()
    ret = []
    grid_W = int((W / cell_sz) + 0.999999)
    grid_H = int((H / cell_sz) + 0.999999)
    grid = np.empty((grid_H,grid_W),  dtype=object)
    for i in np.ndindex(grid.shape): grid[i] = []
    
    # Generate random point
    v = np.random.rand(2) * np.array([H,W])
    queue.append(v)
    ret.append(v)
    x = int(v[1] / cell_sz)
    y = int(v[0] / cell_sz)
    grid[y,x].append(v)

    def new_point(p,min_radius):
        # random point in surrounding between min_radius and 2*min_radius
        r = min_radius * (1+np.random.rand())
        a = 2 * math.pi * np.random.rand()
        return p + np.array([math.sin(a)*r,math.cos(a)*r])

    def has_close_neightbors(p):
        x = int(p[1] / cell_sz)
        y = int(p[0] / cell_sz)
        for ny in range(y-1,y+1+1):
            for nx in range(x-1,x+1+1):
                if (nx >= 0 and nx < len(grid[0]) and ny >= 0 and ny < len(grid)):
                    # Check to see if coordinate is inside the screen
                    if (len(grid[ny][nx]) > 0):
                        for n in grid[ny][nx]:
                            v = p - n
                            d = np.sum(v*v)
                            if (d < min_dist*min_dist):
                                return True;  # There is at least one neighbor who is too close
        return False # No neighbors are too close
    
    # Begin generating other points
    while (len(queue) > 0):
        p = queue.pop()
        for i in range(max_tries):
            new_p = new_point(p, min_dist)
            if (new_p[0] >= 0 and new_p[0] < H and new_p[1] >= 0 and new_p[1] < W and not has_close_neightbors(new_p)):
                queue.append(new_p)
                ret.append(new_p)
                x = int(new_p[1] / cell_sz)
                y = int(new_p[0] / cell_sz)
                grid[y][x].append(new_p)
    
    return np.array(ret)

def weighted_poisson_disk_sampling(img, min_dist=5, max_dist=13, max_tries=40, sigmoidMag=1.5, invert=False):
    cell_sz = max_dist / math.sqrt(2)
    queue = deque()
    ret = []
    W = img.shape[1]
    H = img.shape[0]
    grid_W = int((W / cell_sz) + 0.999999)
    grid_H = int((H / cell_sz) + 0.999999)
    grid = np.empty((grid_H,grid_W),  dtype=object)
    for i in np.ndindex(grid.shape): grid[i] = []
    

    def new_point(p,min_radius):
        # random point in surrounding between min_radius and 2*min_radius
        r = min_radius * (1+np.random.rand())
        a = 2 * math.pi * np.random.rand()
        ret = p + np.array([math.sin(a)*r,math.cos(a)*r])
        return ret if(ret[0]>=0 and ret[0]<H and ret[1]>=0 and ret[1]<W) else None

    def has_close_neightbors(p, mini_d):#=min_dist):
        x = int(p[1] / cell_sz)
        y = int(p[0] / cell_sz)
        for ny in range(y-1,y+1+1):
            for nx in range(x-1,x+1+1):
                if (nx >= 0 and nx < len(grid[0]) and ny >= 0 and ny < len(grid)):
                    # Check to see if coordinate is inside the screen
                    if (len(grid[ny][nx]) > 0):
                        for n in grid[ny][nx]:
                            v = p - n
                            d = np.sum(v*v)
                            if (d < mini_d*mini_d):
                                return True;  # There is at least one neighbor who is too close
        return False # No neighbors are too close


    def sigmoid(x, mag):
        # Given a number between 0 - 1, steer the number towards one pole 
        # Used in this algorthm for adding polarity (dark gets darker, light gets lighter)
        y = 1/(1 + pow(x/(1-x), -mag))
        return y

    # Start off with a uniform distrbution of random points
    # Why we use uniform poisson disc over just randomly generating points: Randomly generated points tend to form tight clusters and way too sparse vacant spaces. This can lead to having high density at lighter regions while having low density at darker regions. Uniform poisson disc ensures a random but still even distribution.
    # Why we don't use only one or very few starting point(s): Imagine starting points begin at the top left of the screen. Since poisson disk in general has a hash RNG component, the points may never reach the bottom right of the screen. This leads to part of the image being fully ignored.
    startPoints = uniform_poisson_disk_sampling(W,H, max_dist, max_tries)
    for v in startPoints:
        queue.append(v)
  
    # # Generate random point
    # v = np.random.rand(2) * np.array([H,W])
    # queue.append(v)
    # ret.append(v)
    # x = int(v[1] / cell_sz)
    # y = int(v[0] / cell_sz)
    # grid[y,x].append(v)

    # Treat every point of the uniform point spread as a source for weighted poisson disk sampling
    #/ Weighted poisson disk sampling priorize tight clusters for dark regions and sparse distibution for lighter regions.
    while (len(queue) > 0):
        p = queue.pop()
        gray = img[int(p[0]),int(p[1])]
        if(invert):
            gray = 255-gray
        minRadius = min_dist + sigmoid(gray/255, sigmoidMag) * (max_dist - min_dist)
        #print(gray, minRadius)
        pointSet = []
        lowGray = gray - 20
        highGray = gray + 20
        for i in range(max_tries):
            new_p = new_point(p, minRadius)
            if(new_p is None):
                continue
            newGray = img[int(new_p[0]),int(new_p[1])]
            if (new_p[0] >= 0 and new_p[0] < H and new_p[1] >= 0 and new_p[1] < W and
                       #lowGray < newGray and newGray < highGray and
                       not has_close_neightbors(new_p, minRadius)):
                queue.append(new_p)
                ret.append(new_p)
                x = int(new_p[1] / cell_sz)
                y = int(new_p[0] / cell_sz)
                grid[y][x].append(new_p)
    return ret


if __name__ == "__main__":
    import cv2

    img = cv2.imread("sources/sting_00.jpg")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    pts = weighted_poisson_disk_sampling(gray, 8, 40, 30, 1.5, False)
    print(len(pts))
    for p in pts:
        cv2.circle(img, (int(p[1]), int(p[0])), 1, (255,0,255),-1)

    cv2.imshow("Poisson disk", img)
    cv2.waitKey()
