import glob
import os
import sys

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
import pygame
import random
import queue
import numpy as np

class SynchronyModel(object):
    def __init__(self):
        self.world,self.init_setting = self._make_setting()
        self.blueprint_library = self.world.get_blueprint_library()
        self.non_player = []
        self.actor_list = []
        self.frame = None
        self.player = None
        self.sensors = []
        self._queues = []
        self._span_player()
        
        self.make_queue(self.world.on_tick)
        for sensor in self.sensors:
            self.make_queue(sensor.listen)
    
        # self.world.apply_settings(self.init_setting)
    
    def make_queue(self, register_event):
        q = queue.Queue()
        register_event(q.put)
        self._queues.append(q)

    def tick(self, timeout):
        self.frame = self.world.tick()
        data = [self._retrieve_data(q, timeout) for q in self._queues[:2]]
        data.append(str(self.player.get_velocity()))
        data.append(self._retrieve_event(self._queues[2], timeout))
        data.append(self._retrieve_event(self._queues[3], timeout))
        data.append(str(self.player.get_transform()))
        
        return data

    def _retrieve_data(self, sensor_queue, timeout):
        while True:
            data = sensor_queue.get(timeout=timeout)
            if data.frame == self.frame:
                return data

    def _retrieve_event(self, sensor_queue, timeout):
        if sensor_queue.empty():
            return False
        while True:
            try:
                data = sensor_queue.get(timeout=timeout)
                if data.frame == self.frame:
                    return data
            except:
                return True

    def _make_setting(self):
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)
        world = client.get_world()
        # synchrony model and fixed time step
        init_setting = world.get_settings()
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.1  # 20 steps  per second
        world.apply_settings(settings)
        return world, init_setting

    def _span_player(self):
        self.my_vehicle_bp = self.world.get_blueprint_library().find('vehicle.tesla.model3')

        self.spawn_points = self.world.get_map().get_spawn_points()
        
        while True:
            random.shuffle(self.spawn_points)
            transform_vehicle = self.spawn_points[0]
            try:
                my_vehicle = self.world.spawn_actor(self.my_vehicle_bp, transform_vehicle)
                break
            except RuntimeError:
                continue
        
        self._span_sensor(my_vehicle)
        self.actor_list.append(my_vehicle)
        self.player = my_vehicle

    def _span_sensor(self, player):
        camera_bp = self.blueprint_library.find('sensor.camera.rgb')
        # camera_d_bp = self.blueprint_library.find('sensor.camera.depth')
        # lidar_bp = self.blueprint_library.find('sensor.lidar.ray_cast')
        # imu_bp = self.blueprint_library.find('sensor.other.imu')
        lane_bp = self.blueprint_library.find('sensor.other.lane_invasion')
        col_bp = self.blueprint_library.find('sensor.other.collision')

        camera_bp.set_attribute('image_size_x', str(640))
        camera_bp.set_attribute('image_size_y', str(480))
        camera_bp.set_attribute('fov', '120')

        # camera_d_bp.set_attribute('image_size_x', str(1920))
        # camera_d_bp.set_attribute('image_size_y', str(1080))
        # camera_d_bp.set_attribute('fov', '90')

        # lidar_bp.set_attribute('rotation_frequency', '20')

        transform_sensor = carla.Transform(carla.Location(x=2.3, y=0, z=1))

        my_camera = self.world.spawn_actor(camera_bp, transform_sensor, attach_to=player)
        # my_camera_d = self.world.spawn_actor(camera_d_bp, transform_sensor, attach_to=player)
        # my_lidar = self.world.spawn_actor(lidar_bp, transform_sensor, attach_to=player)
        # my_imu = self.world.spawn_actor(imu_bp, transform_sensor, attach_to=player)
        my_lane = self.world.spawn_actor(lane_bp, transform_sensor, attach_to=player)
        my_col = self.world.spawn_actor(col_bp, transform_sensor, attach_to=player)

        self.actor_list.append(my_camera)
        # self.actor_list.append(my_camera_d)
        # self.actor_list.append(my_lidar)
        # self.actor_list.append(my_imu)
        self.actor_list.append(my_lane)
        self.actor_list.append(my_col)

        self.sensors.append(my_camera)
        # self.sensors.append(my_camera_d)
        # self.sensors.append(my_lidar)
        # self.sensors.append(my_imu)
        self.sensors.append(my_lane)
        self.sensors.append(my_col)

    def _reset_player(self):
        self.actor_list = []
        self.sensors = []
        
        while True:
            random.shuffle(self.spawn_points)
            transform_vehicle = self.spawn_points[0]
            try:
                my_vehicle = self.world.spawn_actor(self.my_vehicle_bp, transform_vehicle)
                break
            except RuntimeError:
                continue
        
        self._span_sensor(my_vehicle)
        self.actor_list.append(my_vehicle)
        self.player = my_vehicle

        self._queues = []
        self.make_queue(self.world.on_tick)
        for sensor in self.sensors:
            self.make_queue(sensor.listen)


def process_image(image):
    array = np.frombuffer(image.raw_data, dtype=np.dtype('uint8'))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    # return array.astype(np.float32)
    return array.copy()

def draw_image(surface, array, blend=False):
    image_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
    if blend:
        image_surface.set_alpha(100)
    surface.blit(image_surface, (0, 0))

def process_imu(imu):
    accelerometer_data = str(imu.accelerometer)
    gyroscope_data = str(imu.gyroscope)

    acc_x_pos, acc_y_pos, acc_z_pos = accelerometer_data.index('x'), accelerometer_data.index('y'), accelerometer_data.index('z')
    acc_x, acc_y, acc_z = float(accelerometer_data[acc_x_pos + 2:acc_y_pos - 2]), float(accelerometer_data[acc_y_pos + 2:acc_z_pos - 2]), float(accelerometer_data[acc_z_pos + 2:-1])

    gyro_x_pos, gyro_y_pos, gyro_z_pos = gyroscope_data.index('x'), gyroscope_data.index('y'), gyroscope_data.index('z')
    gyro_x, gyro_y, gyro_z = float(gyroscope_data[gyro_x_pos + 2:gyro_y_pos - 2]), float(gyroscope_data[gyro_y_pos + 2:gyro_z_pos - 2]), float(gyroscope_data[gyro_z_pos + 2:-1])

    return np.array([acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, imu.compass])

def process_vel(vel):
    x_pos, y_pos, z_pos = vel.index('x'), vel.index('y'), vel.index('z')
    x, y, z = float(vel[x_pos + 2:y_pos - 2]), float(vel[y_pos + 2:z_pos - 2]), float(vel[z_pos + 2:-1])
    return np.array([x, y, z])

def process_pos(pos):
    x_pos, y_pos, z_pos = pos.index('x'), pos.index('y'), pos.index('z')
    x, y = float(pos[x_pos + 2:y_pos - 2]), float(pos[y_pos + 2:z_pos - 2])
    
    yaw_pos, roll_pos = pos.index('yaw'), pos.index('roll')
    yaw = float(pos[yaw_pos + 4:roll_pos-2])
    # return np.array([x, y, yaw])
    return [x, y, yaw]

def process_goal(pos):
    x_pos, y_pos, z_pos = pos.index('x'), pos.index('y'), pos.index('z')
    x, y = float(pos[x_pos + 2:y_pos - 2]), float(pos[y_pos + 2:z_pos - 2])
    return [x, y]

def should_quit():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return True
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_ESCAPE:
                return True
    return False


def get_font():
    fonts = [x for x in pygame.font.get_fonts()]
    default_font = 'ubuntumono'
    font = default_font if default_font in fonts else fonts[0]
    font = pygame.font.match_font(font)
    return pygame.font.Font(font, 14)


class Agent(object):
    def __init__(self):
        pygame.init()
        self.display = pygame.display.set_mode(
            (640, 480),
            pygame.HWSURFACE | pygame.DOUBLEBUF)
        self.font = get_font()
        self.clock = pygame.time.Clock()
        # with SynchronyModel() as sync_mode:
        #     self.sync_mode = sync_mode
        self.sync_mode = SynchronyModel()
        self.waypoint = self.sync_mode.world.get_map().get_waypoint(self.sync_mode.player.get_transform().location)
    
    def reset(self):
        print('resetting agent...')
        for actor in self.sync_mode.actor_list:
            actor.destroy()
        self.sync_mode._reset_player()
        # self.waypoint = self.sync_mode.world.get_map().get_waypoint(self.sync_mode.player.get_transform().location)
        print('done\n')

    def step(self, action):
        if should_quit():
            print('destroying actors.')
            for actor in self.sync_mode.actor_list:
                actor.destroy()
            pygame.quit()
            print('done.')
            return None

        control = carla.VehicleControl(steer = action[0], throttle = action[1], brake = action[2])
        self.sync_mode.player.apply_control(control)
        # print('curr: ', self.sync_mode.player.get_transform())
        # self.waypoint = random.choice(self.waypoint.next(1.5))
        # self.sync_mode.player.set_transform(self.waypoint.transform)

        self.clock.tick()
        # snapshot, image_rgb, image_depth, point_cloud, imu_data = self.sync_mode.tick(timeout=2.0)
        snapshot, image_rgb, vel_data, lane, coll, pos_data = self.sync_mode.tick(timeout=2.0)
        # print(imu_data)
        
        image_rgb = process_image(image_rgb)
        # imu_data = process_imu(imu_data)
        vel_data = process_vel(vel_data)
        pos_data = process_pos(pos_data)
        
        
        draw_image(self.display, image_rgb)
        # display frequency
        # fps = round(1.0 / snapshot.timestamp.delta_seconds)
        # self.display.blit(
        #     self.font.render('% 5d FPS (real)' % self.clock.get_fps(), True, (255, 255, 255)),
        #     (8, 10))
        # self.display.blit(
        #     self.font.render('% 5d FPS (simulated)' % fps, True, (255, 255, 255)),
        #     (8, 28))
        pygame.display.flip()
        
        if coll:
            done = True
        else:
            done = False

        return image_rgb, vel_data, lane, coll, pos_data, done

    def close(self):
        print('destroying actors.')
        for actor in self.sync_mode.actor_list:
            actor.destroy()
        pygame.quit()
        print('done.')

    def reward(self, lane, coll, vel):
        r = 0
        if lane:
            r = -10
        if coll:
            r = -100
            return r
        r += sum([i**2 for i in vel])/100
        return r
    
    def set_goal(self, distance):
        # print('curr: ', self.sync_mode.player.get_transform())
        self.waypoint = self.sync_mode.world.get_map().get_waypoint(self.sync_mode.player.get_transform().location)
        # self.sync_mode.player.set_transform(self.waypoint.transform)

        # self.clock.tick()
        # snapshot, image_rgb, vel_data, lane, coll, pos_data = self.sync_mode.tick(timeout=2.0)
        
        # image_rgb = process_image(image_rgb)
        # draw_image(self.display, image_rgb)
        # pygame.display.flip()
        
        # print('curr: ', self.sync_mode.player.get_transform())
        
        self.waypoint = random.choice(self.waypoint.next(distance))
        
        # print('next: ', self.waypoint.transform.location)

        
        # self.clock.tick()
        # snapshot, image_rgb, vel_data, lane, coll, pos_data = self.sync_mode.tick(timeout=2.0)
        return process_goal(str(self.waypoint.transform.location))
        

def main():
    action = [0.0, 0.0, 0.0]
    agent = Agent()
    agent.reset()
    while True:
        image, vel, lane, coll, pos, done = agent.step(action)
        if vel[2] == 0.0:
            break
    goal = agent.set_goal(10)    
    for i in range(20):
        image, vel, lane, coll, pos, done = agent.step([-0.010, 0.453, 0.003])
        print(vel)
    agent.close()


if __name__ == '__main__':
    main()
