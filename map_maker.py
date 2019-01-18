import numpy as np
from random import shuffle

#def make_map(width, length, obstacles):
def make_map(width=10,length=20,obstacles=['F','L']*32):
    num_blanks = width*length - len(obstacles)
    obstacles += [' ']*num_blanks
    shuffle(obstacles)
    # Add goal row
    goal_row = [' ']*(width-1)+['G']
    shuffle(goal_row)
    level = obstacles + goal_row
    # Conver to numpy array
    level_np = np.array(obstacles)
    level_np_rs = level_np.reshape((length, width))

    # front/back wall
    fbw = np.array(['*']*(width+2))
    # Last row (Goal Row)
    lr = np.insert(np.array([' ']*(width-1)), 0, np.array('G'), 0)
    np.random.shuffle(lr)
    # side walls
    sw = np.array(['*']*(length+1))
    # newline walls
    nlw = np.array(['\n']*(length+3))

    # Insert goal row
    level_np_rs = np.insert(level_np_rs, length, lr, 0)
    # Insert side walls
    level_np_rs = np.insert(level_np_rs, 0, sw, 1)
    level_np_rs = np.insert(level_np_rs, width+1, sw, 1)
    # Insert back wall
    level_np_rs = np.insert(level_np_rs, 0, fbw, 0)
    # Insert front wall
    level_np_rs = np.insert(level_np_rs, length+2, fbw, 0)
    level_np_rs = np.insert(level_np_rs, width+2, nlw, 1)

    return level_np_rs 
    #return level_np_rs.reshape((-1)) 

#print(make_map(100,200,['A','F','L','S']*80))
#print([str(list(make_map()))])
m = make_map()
for i in range(m.shape[0]):
    for j in range(m.shape[1]):
        print(m[i,j], end='')


