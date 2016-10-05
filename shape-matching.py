import math
import time
import numpy as np
from numpy.linalg import inv
from scipy.linalg import sqrtm
from numpy.linalg import eig
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Polygon
import matplotlib.animation as animation


#################################
# GRAPH REST SHAPE w/ Polyfill
#################################

def createCircle(cx, cy,r, num_segments): 
    verts= np.zeros((num_segments,2))
    theta = (2 * 3.1415926) / num_segments
    c = math.cos(theta)
    s = math.sin(theta)
    x = r
    y = 0 
    for i in range(num_segments):
        verts[i,0] = x + cx
        verts[i,1] = y + cy
        t = x;
        x = c * x - s * y
        y = s * t + c * y
    return verts

class ShapeMatching:
    def __init__(self, x, y, r, n):
        self.n = n
        self.initial_pos = createCircle(x, y, r, self.n)
        self.curr_pos = self.initial_pos
        self.curr_vel = np.zeros((self.n,2))
        self.goal_pos = self.initial_pos
        self.mass = 1.0
        self.alpha = 0.4
        self.dt = 1.0 / 60.0
        self.dt_inv = 1.0 / self.dt
        self.gravity = np.array([0.0,-9.8])
        self.initial_com = self.calcCenterOfMass(self.initial_pos)
        self.initial_rel_pos = self.initial_pos - self.initial_com  
        self.curr_com = self.initial_com
        self.curr_rel_pos = self.initial_rel_pos
        
    def calcCenterOfMass(self, pos):
        sum = np.array([0.0,0.0])
        for p in pos:
            sum += p
        sum /= pos.shape[0]
        return sum

    def calcA_qq(self, q_i):
        sum = np.zeros((2,2))
        for q in q_i:
            sum += np.outer(q,np.transpose(q))
        return inv(sum)

    def calcA_pq(self, p_i, q_i):
        sum = np.zeros((2,2))
        for i in range(p_i.shape[0]):
            sum += np.outer(p_i[i],np.transpose(q_i[i]))
        return sum

    def calcR(self, A_pq):
        S = sqrtm(np.dot(np.transpose(A_pq),A_pq))
        R = np.dot(A_pq,inv(S))
        return R

    def calcGoalPositions(self, R, q_i, curr_com):
        goal = np.zeros((q_i.shape[0],2))
        for i in range(q_i.shape[0]):
            goal[i] = R.dot(q_i[i]) + curr_com
        return goal
    
    def collisionDetection(self, curr_pos, curr_vel):
        for i in range(curr_pos.shape[0]):
            if curr_pos[i,1] < 0.0:
                curr_pos[i,1] = 0.0
                curr_vel[i] = np.array([0.0,0.0])
        return curr_pos, curr_vel

    def integrate(self, pos, curr_pos, curr_vel, goal_pos):
        for i in range(curr_pos.shape[0]):
            curr_vel[i] = curr_vel[i] + self.alpha * (goal_pos[i] - pos[i]) * self.dt_inv + self.dt * self.gravity
            curr_pos[i] = curr_pos[i] + self.dt * curr_vel[i]
        return curr_pos, curr_vel
        
    def step(self):
        vel = self.curr_vel + self.dt * self.gravity
        pos = self.curr_pos + self.dt * vel
        self.curr_com = self.calcCenterOfMass(pos)
        self.curr_rel_pos = pos - self.curr_com
        A_pq = self.calcA_pq(self.curr_rel_pos, self.initial_rel_pos)
        R = self.calcR(A_pq)
        self.goal_pos = self.calcGoalPositions(R, self.initial_rel_pos, self.curr_com)
        self.curr_pos, self.curr_vel = self.integrate(pos, self.curr_pos, self.curr_vel, self.goal_pos)
        self.curr_pos, self.curr_vel = self.collisionDetection(self.curr_pos, self.curr_vel)

sm = ShapeMatching(0.0, 5.5, 2.0, 20)

#################################
# ANIMATION
#################################

if True:
    # initialization function: plot the background of each frame
    def init():
        curr_pos.set_data([], [])
        goal_pos.set_data([], [])
        curr_com.set_data([], [])
        return curr_pos, goal_pos, curr_com

    # animation function. This is called sequentially
    def animate(i):
        sm.step()
        curr_pos.set_data(sm.curr_pos[:,0], sm.curr_pos[:,1])
        goal_pos.set_data(sm.goal_pos[:,0], sm.goal_pos[:,1])
        curr_com.set_data(sm.curr_com[0], sm.curr_com[1])
        patch.set_xy(sm.curr_pos)
        return curr_pos, goal_pos, curr_com

    
    fig, ax = plt.subplots()
    curr_pos, = ax.plot(sm.curr_pos[:,0], sm.curr_pos[:,1],'ro')
    goal_pos, = ax.plot(sm.goal_pos[:,0], sm.goal_pos[:,1],'bx')
    curr_com, = ax.plot(sm.curr_com[0], sm.curr_com[1],'bo')
    patch = plt.Polygon(np.zeros((sm.n,2)), color='b', alpha=0.25)
    ax.add_patch(patch)

    # call the animator. blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=250, interval=50, blit=False)

    plt.xlim(-5,5)
    plt.ylim(-1,20)
    plt.axhspan(0.0, -1.0, facecolor='0.5', alpha=0.5)
    plt.gca().set_aspect('equal', adjustable='box')

    #anim.save('shape-matching.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
    plt.show()


#################################
# STATIC PLOT
#################################

if False:
    fig, ax = plt.subplots()
    ax.add_patch(Polygon(sm.curr_pos, True, alpha=0.4, zorder=1))
    plt.scatter(sm.initial_pos[:,0], sm.initial_pos[:,1], c='g',marker = 'o', s=120, zorder=2)
    plt.scatter(sm.curr_pos[:,0], sm.curr_pos[:,1], c='r',marker = 'o', s=120, zorder=2)
    plt.scatter(sm.goal_pos[:,0], sm.goal_pos[:,1], c='b',marker = 'o', s=120, zorder=2)
    plt.xlim(-3, 3)
    plt.ylim(-1, 6)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

