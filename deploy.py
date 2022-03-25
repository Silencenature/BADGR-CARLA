def main():
    import numpy as np
    from envs.simulator import Agent
    from Plan.action_plan import plan
    # import os
    import torch

    # os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

    use_cuda = torch.cuda.is_available()
    device   = torch.device("cuda" if use_cuda else "cpu")

    max_steps = 20
    N = 4096
    
    planner = plan('ep-95.pth', N)
    
    agent = Agent()
    agent.reset()
    
    action = [0.0, 0.0, 0.0]
    while True:
        image, vel, lane, coll, pos, done = agent.step(action)
        if vel[2] == 0.0:
            break
        
    goal = agent.set_goal(10)
    goal = torch.Tensor(goal).to(device)
    action_serise = np.zeros((10, 3))
    step = 0
    print('start moving...')
    while step < max_steps:
        try:
            action_serise = planner.update_actions(image, vel, pos, action_serise, goal)
            if vel[0] == 0 and vel[1] == 0:
                action_serise[:,2] = 0
                print('step: %s, actions: %.3f, %.3f, %.3f' % (step,action_serise[0][0],action_serise[0][1],action_serise[0][2]))
                for _ in range(20):
                    image, vel, lane, coll, pos, done = agent.step(action_serise[0].tolist())
                step += 1
            else:      
                print('step: %s, actions: %.3f, %.3f, %.3f' % (step,action_serise[0][0],action_serise[0][1],action_serise[0][2]))
                for _ in range(10):
                    image, vel, lane, coll, pos, done = agent.step(action_serise[0].tolist())
                step += 1
            if done:
                break
        except:
            agent.close()
            return
    
    print('\nfinish in %.1fs' % (step*0.1))
    if done:
        print('collide')
    else:
        print('not collide')
    print('goal: %.2f,%.2f, final position: %.2f,%.2f' % (goal[0], goal[1], pos[0], pos[1]))
    diff_x, diff_y = goal[0]-pos[0], goal[1]-pos[1]
    print('difference if %.2f,%.2f, distance is %.2f' % (diff_x,diff_y,(diff_x**2+diff_y**2)**0.5))
    
    agent.close()

    return

if __name__ == "__main__":
    main()