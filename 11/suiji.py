import glob
import os
import sys
import math
import cv2
import time
import random
import numpy as np
import skfuzzy as fuzz
import skfuzzy.control as ctrl
import matplotlib.pyplot as plt
from collections import deque

from carla import Vector3D

ROUNDING_FACTOR = 2
SECONDS_PER_EPISODE = 1.5
LIMIT_RADAR = 500
np.random.seed(32)
random.seed(32)
MAX_LEN = 500

try:
    sys.path.append(glob.glob('../../../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla


class CarlaVehicle():

    def __init__(self):
        self.client = carla.Client(host='127.0.0.1', port=2000)
        self.client.set_timeout(5.0)
        self.radar_data = deque(maxlen=MAX_LEN)
        self.actor_list = []

    def reset(self, loc):
        self.world = self.client.get_world()
        self.collision_hist = []
        self.actor_list = []
        self.barrier_list = []
        self.map = self.world.get_map()
        print('reset')
        self.blueprint_library = self.world.get_blueprint_library()
        self.model_3 = self.blueprint_library.filter("model3")[0]
        self.model_4 = self.blueprint_library.filter("model3")[0]
        self.model_5 = self.blueprint_library.filter("model3")[0]
        self.static_prop = self.blueprint_library.filter("static.prop.streetbarrier")[0]
        # self.a = np.random.uniform(50, 40)
        # self.b = np.random.uniform(45, 25)
        # self.c = np.random.uniform(35, 25)
        # self.e = np.random.uniform(20, 5)
        # self.f = np.random.uniform(15, 5)
        # self.a = np.random.uniform(25, 23)
        # self.b = np.random.uniform(25, 23)
        # self.c = np.random.uniform(15, 14)
        self.e = np.random.uniform(-65, -75)
        self.f = np.random.uniform(-65, -75)
        # self.stay_p = np.random.uniform(25, 20)
        # self.change_p = np.random.uniform(25,20)
        # self.a1 = np.random.uniform(0.6, 0.2)
        # self.a2 = np.random.uniform(0.7, 0.5)
        # self.a3 = np.random.uniform(0.4, 0.2)
        self.a5 = np.random.uniform(0.2, 0)
        self.a6 = np.random.uniform(0.1, 0)
        self.lane_id_ego = 0
        self.lane_id_target = 0
        self.yaw_vehicle = 0
        self.yaw_target_road = 0
        init_loc = carla.Location(x=-25, y=132, z=1)
        target_waypoint1 = carla.Location(x=-15, y=140, z=1)
        target_waypoint2 = carla.Location(x=-10, y=132, z=1)
        prop_lt_f1_loc = carla.Transform(carla.Location(x=-15, y=132, z=1),
                                         carla.Rotation(pitch=0.094270, yaw=0, roll=-1.246277))
        prop_lt_f2_loc = carla.Transform(carla.Location(x=-13, y=140, z=1),
                                         carla.Rotation(pitch=0.094270, yaw=0, roll=-1.246277))
        prop_lt_ego_loc = carla.Transform(init_loc, carla.Rotation(pitch=0.094270, yaw=0, roll=-1.246277))
        prop_lt_b1_loc = carla.Transform(carla.Location(x=self.e, y=132, z=1),
                                         carla.Rotation(pitch=0.094270, yaw=0, roll=-1.246277))
        prop_lt_b2_loc = carla.Transform(carla.Location(x=self.f, y=140, z=1),
                                         carla.Rotation(pitch=0.094270, yaw=0, roll=-1.246277))
        self.prop_lt_f1 = self.world.spawn_actor(self.model_4, prop_lt_f1_loc)
        self.prop_lt_f2 = self.world.spawn_actor(self.model_3, prop_lt_f2_loc)
        self.prop_lt_ego = self.world.spawn_actor(self.model_5, prop_lt_ego_loc)
        prop_lt_b1 = self.world.spawn_actor(self.model_3, prop_lt_b1_loc)
        prop_lt_b2 = self.world.spawn_actor(self.model_3, prop_lt_b2_loc)
        self.prop_lt_ego.set_target_velocity(Vector3D(x=10, y=0, z=0))
        self.prop_lt_f1.set_target_velocity(Vector3D(x=7, y=0, z=0))
        self.prop_lt_f2.set_target_velocity(Vector3D(x=13, y=0, z=0))
        self.prop_lt_f1.apply_control(carla.VehicleControl(throttle=0.05, steer=0))
        self.prop_lt_f2.apply_control(carla.VehicleControl(throttle=0.1, steer=0))
        prop_lt_b1.apply_control(carla.VehicleControl(throttle=self.a5, steer=0))
        prop_lt_b2.apply_control(carla.VehicleControl(throttle=self.a6, steer=0))
        self.actor_list.append(self.prop_lt_f1)
        self.actor_list.append(self.prop_lt_f2)
        self.actor_list.append(self.prop_lt_ego)
        self.actor_list.append(prop_lt_b1)
        self.actor_list.append(prop_lt_b2)
        self.next_lane_target = self.map.get_waypoint(init_loc).get_right_lane()
        prop_lt_1_loc = carla.Transform(carla.Location(x=-15, y=128, z=0.014589),
                                        carla.Rotation(pitch=0.145088, yaw=-0.817204, roll=0.000000))
        prop_lt_2_loc = carla.Transform(carla.Location(x=-18, y=128, z=0.009964),
                                        carla.Rotation(pitch=0.119906, yaw=-0.817204, roll=0.000000))
        prop_lt_3_loc = carla.Transform(carla.Location(x=-21, y=128, z=0.009964),
                                        carla.Rotation(pitch=0.119906, yaw=-0.817204, roll=0.000000))
        prop_lt_4_loc = carla.Transform(carla.Location(x=-24, y=128, z=0.009964),
                                        carla.Rotation(pitch=0.119906, yaw=-0.817204, roll=0.000000))
        prop_lt_5_loc = carla.Transform(carla.Location(x=-27, y=128, z=0.009964),
                                        carla.Rotation(pitch=0.119906, yaw=-0.817204, roll=0.000000))

        self.prop_lt_1 = self.world.spawn_actor(self.static_prop, prop_lt_1_loc)
        self.prop_lt_2 = self.world.spawn_actor(self.static_prop, prop_lt_2_loc)
        self.prop_lt_3 = self.world.spawn_actor(self.static_prop, prop_lt_3_loc)
        self.prop_lt_4 = self.world.spawn_actor(self.static_prop, prop_lt_4_loc)
        self.prop_lt_5 = self.world.spawn_actor(self.static_prop, prop_lt_5_loc)

        prop_end_rt_loc = carla.Transform(carla.Location(x=-29, y=130, z=0.014589),
                                          carla.Rotation(pitch=0.145088, yaw=-0.817204 + 90, roll=0.000000))
        prop_end_lt_loc = carla.Transform(carla.Location(x=-29, y=132, z=0.009964),
                                          carla.Rotation(pitch=0.119906, yaw=-0.817204 + 90, roll=0.000000))
        prop_start_lt_loc = carla.Transform(carla.Location(x=-29, y=134, z=0.009964),
                                            carla.Rotation(pitch=0.119906, yaw=-0.817204 + 90, roll=0.000000))
        self.prop_end_rt = self.world.spawn_actor(self.static_prop, prop_end_rt_loc)
        self.prop_end_lt = self.world.spawn_actor(self.static_prop, prop_end_lt_loc)
        self.prop_start_lt = self.world.spawn_actor(self.static_prop, prop_start_lt_loc)
        prop_end_rt_loc1 = carla.Transform(carla.Location(x=-18, y=137, z=0.014589),
                                           carla.Rotation(pitch=0.145088, yaw=-0.817204 + 90, roll=0.000000))
        prop_end_lt_loc2 = carla.Transform(carla.Location(x=-18, y=141, z=0.009964),
                                           carla.Rotation(pitch=0.119906, yaw=-0.817204 + 90, roll=0.000000))
        prop_start_lt_loc3 = carla.Transform(carla.Location(x=-18, y=139, z=0.009964),
                                             carla.Rotation(pitch=0.119906, yaw=-0.817204 + 90, roll=0.000000))
        self.prop_end_rt1 = self.world.spawn_actor(self.static_prop, prop_end_rt_loc1)
        self.prop_end_lt2 = self.world.spawn_actor(self.static_prop, prop_end_lt_loc2)
        self.prop_start_lt3 = self.world.spawn_actor(self.static_prop, prop_start_lt_loc3)
        prop_rt_1_loc = carla.Transform(carla.Location(x=-27, y=135, z=0.009964),
                                        carla.Rotation(pitch=0.119906, yaw=-0.817204, roll=0.000000))
        prop_rt_2_loc = carla.Transform(carla.Location(x=-24, y=135, z=0.009964),
                                        carla.Rotation(pitch=0.119906, yaw=-0.817204, roll=0.000000))
        prop_rt_3_loc = carla.Transform(carla.Location(x=-21, y=135, z=0.009964),
                                        carla.Rotation(pitch=0.119906, yaw=-0.817204, roll=0.000000))
        prop_rt_4_loc = carla.Transform(carla.Location(x=-18, y=135, z=0.009964),
                                        carla.Rotation(pitch=0.119906, yaw=-0.817204, roll=0.000000))
        prop_rt_1x_loc = carla.Transform(carla.Location(x=-9, y=128, z=0.009964),
                                         carla.Rotation(pitch=0.119906, yaw=-0.817204, roll=0.000000))
        prop_rt_2x_loc = carla.Transform(carla.Location(x=-6, y=128, z=0.009964),
                                         carla.Rotation(pitch=0.119906, yaw=-0.817204, roll=0.000000))
        prop_rt_3x_loc = carla.Transform(carla.Location(x=-3, y=128, z=0.009964),
                                         carla.Rotation(pitch=0.119906, yaw=-0.817204, roll=0.000000))
        prop_rt_4x_loc = carla.Transform(carla.Location(x=0, y=128, z=0.009964),
                                         carla.Rotation(pitch=0.119906, yaw=-0.817204, roll=0.000000))
        self.prop_rt_1x = self.world.spawn_actor(self.static_prop, prop_rt_1x_loc)
        self.prop_rt_2x = self.world.spawn_actor(self.static_prop, prop_rt_2x_loc)
        self.prop_rt_3x = self.world.spawn_actor(self.static_prop, prop_rt_3x_loc)
        self.prop_rt_4x = self.world.spawn_actor(self.static_prop, prop_rt_4x_loc)
        self.prop_rt_1 = self.world.spawn_actor(self.static_prop, prop_rt_1_loc)
        self.prop_rt_2 = self.world.spawn_actor(self.static_prop, prop_rt_2_loc)
        self.prop_rt_3 = self.world.spawn_actor(self.static_prop, prop_rt_3_loc)
        self.prop_rt_4 = self.world.spawn_actor(self.static_prop, prop_rt_4_loc)
        prop_rt_5_loc = carla.Transform(carla.Location(x=-3, y=143, z=0.009964),
                                        carla.Rotation(pitch=0.119906, yaw=-0.817204, roll=0.000000))
        prop_rt_6_loc = carla.Transform(carla.Location(x=-15, y=143, z=0.009964),
                                        carla.Rotation(pitch=0.119906, yaw=-0.817204, roll=0.000000))
        prop_rt_7_loc = carla.Transform(carla.Location(x=-12, y=143, z=0.009964),
                                        carla.Rotation(pitch=0.119906, yaw=-0.817204, roll=0.000000))
        prop_rt_8_loc = carla.Transform(carla.Location(x=-9, y=143, z=0.009964),
                                        carla.Rotation(pitch=0.119906, yaw=-0.817204, roll=0.000000))
        prop_rt_9_loc = carla.Transform(carla.Location(x=-6, y=143, z=0.009964),
                                        carla.Rotation(pitch=0.119906, yaw=-0.817204, roll=0.000000))
        self.prop_rt_7 = self.world.spawn_actor(self.static_prop, prop_rt_7_loc)
        self.prop_rt_8 = self.world.spawn_actor(self.static_prop, prop_rt_8_loc)
        self.prop_rt_9 = self.world.spawn_actor(self.static_prop, prop_rt_9_loc)
        self.prop_rt_5 = self.world.spawn_actor(self.static_prop, prop_rt_5_loc)
        self.prop_rt_6 = self.world.spawn_actor(self.static_prop, prop_rt_6_loc)
        self.actor_list.append(self.prop_lt_1)
        self.actor_list.append(self.prop_lt_2)
        self.actor_list.append(self.prop_lt_3)
        self.actor_list.append(self.prop_lt_4)
        self.actor_list.append(self.prop_lt_5)
        self.actor_list.append(self.prop_end_lt)
        self.actor_list.append(self.prop_end_rt)
        self.actor_list.append(self.prop_start_lt)
        self.actor_list.append(self.prop_end_rt1)
        self.actor_list.append(self.prop_end_lt2)
        self.actor_list.append(self.prop_start_lt3)
        self.actor_list.append(self.prop_rt_1x)
        self.actor_list.append(self.prop_rt_2x)
        self.actor_list.append(self.prop_rt_3x)
        self.actor_list.append(self.prop_rt_4x)
        self.actor_list.append(self.prop_rt_1)
        self.actor_list.append(self.prop_rt_2)
        self.actor_list.append(self.prop_rt_3)
        self.actor_list.append(self.prop_rt_4)
        self.actor_list.append(self.prop_rt_5)
        self.actor_list.append(self.prop_rt_6)
        self.actor_list.append(self.prop_rt_7)
        self.actor_list.append(self.prop_rt_8)
        self.actor_list.append(self.prop_rt_9)

        transform = carla.Transform(carla.Location(x=2.5, z=0.7))
        colsensor = self.blueprint_library.find("sensor.other.collision")
        self.colsensor = self.world.spawn_actor(colsensor, transform, attach_to=self.prop_lt_ego)
        self.actor_list.append(self.colsensor)
        self.colsensor.listen(lambda event: self.collision_data(event))
        self.episode_start = time.time()
        transform = carla.Transform(carla.Location(x=2.5, z=0.7))
        self.radar = self.blueprint_library.find('sensor.other.radar')
        self.radar.set_attribute("range", f"80")
        self.radar.set_attribute("horizontal_fov", f"60")
        self.radar.set_attribute("vertical_fov", f"25")
        self.resetRadarData(80, 60, 25)
        self.sensor = self.world.spawn_actor(self.radar, transform, attach_to=self.prop_lt_ego)
        self.actor_list.append(self.sensor)
        self.sensor.listen(lambda data: self.process_radar(data))
        data = np.array(self.radar_data)
        return data[-LIMIT_RADAR:]

    def resetRadarData(self, dist, hfov, vfov):
        alt = 2 * math.pi / vfov
        azi = 2 * math.pi / hfov
        vel = 0
        deque_list = []
        for _ in range(MAX_LEN // 4):
            altitude = random.uniform(-alt, alt)
            deque_list.append(altitude)
            azimuth = random.uniform(-azi, azi)
            deque_list.append(azimuth)
            distance = random.uniform(10, dist)
            deque_list.append(distance)
            deque_list.append(vel)
        self.radar_data.extend(deque_list)

    def process_radar(self, radar):
        self._Radar_callback_plot(radar)
        points = np.frombuffer(buffer=radar.raw_data, dtype='f4')
        points = np.reshape(points, (len(radar), 4))
        for i in range(len(radar)):
            self.radar_data.append(points[i, 0])
            self.radar_data.append(points[i, 1])
            self.radar_data.append(points[i, 2])
            self.radar_data.append(points[i, 3])

    def _Radar_callback_plot(self, radar_data):
        current_rot = radar_data.transform.rotation
        velocity_range = 7.5  # m/s
        world = self.world
        debug = world.debug

        def clamp(min_v, max_v, value):
            return max(min_v, min(value, max_v))

        for detect in radar_data:
            azi = math.degrees(detect.azimuth)
            alt = math.degrees(detect.altitude)
            fw_vec = carla.Vector3D(x=detect.depth - 0.25)
            carla.Transform(carla.Location(), carla.Rotation(pitch=current_rot.pitch + alt, yaw=current_rot.yaw + azi,
                                                             roll=current_rot.roll)).transform(fw_vec)
            norm_velocity = detect.velocity / velocity_range
            r = int(clamp(0.0, 1.0, 1.0 - norm_velocity) * 255.0)
            g = int(clamp(0.0, 1.0, 1.0 - abs(norm_velocity)) * 255.0)
            b = int(abs(clamp(- 1.0, 0.0, - 1.0 - norm_velocity)) * 255.0)
            debug.draw_point(radar_data.transform.location + fw_vec, size=0.075, life_time=0.06, persistent_lines=False,
                             color=carla.Color(r, g, b))

    def collision_data(self, event):
        self.collision_hist.append(event)

    def step(self, action):
        self.model_5.apply_control(
            carla.VehicleControl(throttle=0.1, steer=action[1], brake=action[2], reverse=action[3]))

    def destroy(self):
        print('destroying actors')
        for actor in self.actor_list:
            actor.destroy()
        print('done')




    def FSM(self, action):
        class LaneChangeFSM:
            def __init__(self):
                self.states = {
                    'LANE_KEEPING': self.lane_keeping,
                    'LANE_CHANGING': self.lane_changing,
                    'LANE_CHANGE_PREPARE': self.lane_change_prepare
                }
                self.current_state = 'LANE_KEEPING'

            def lane_keeping(self, sensors):

                if sensors['left_space'] > 10 and sensors['right_space'] > 10 and sensors['car_ahead_speed'] < 40:
                    self.current_state = 'LANE_CHANGE_PREPARE'
                    print("Preparing to change lane")
                else:
                    print("Keeping current lane")
                return self.current_state

            def lane_change_prepare(self, sensors):

                if sensors['blind_spot_left'] == False and sensors['blind_spot_right'] == False:
                    self.current_state = 'LANE_CHANGING'
                    print("Changing lane")
                else:
                    self.current_state = 'LANE_KEEPING'
                    print("Cannot change lane, blind spot occupied")
                return self.current_state

            def lane_changing(self, sensors):
                if sensors['lane_changed']:
                    self.current_state = 'LANE_KEEPING'
                    print("Lane changed successfully")
                else:
                    print("Still changing lane")
                return self.current_state

            def update(self, sensors):

                self.current_state = self.states[self.current_state](sensors)


        def get_sensor_data():
            return {
                'left_space': 15,
                'right_space': 12,
                'car_ahead_speed': 30,
                'blind_spot_left': False,
                'blind_spot_right': False,
                'lane_changed': False
            }

        fsm = LaneChangeFSM()
        for _ in range(10):
            sensors = get_sensor_data()
            fsm.update(sensors)
        done = False
        action[0] = 0.2
        action[1] = 0
        self.prop_lt_ego.apply_control(carla.VehicleControl(throttle=action[0], steer=action[1], brake=action[2], reverse=action[3]))
        self.collision_list = []
        self.lane_id_ego = self.map.get_waypoint(self.prop_lt_ego.get_location()).lane_id
        self.lane_id_target = self.next_lane_target.lane_id
        current_location_ego = self.prop_lt_ego.get_location()
        current_location_f1 = self.prop_lt_f1.get_location()
        current_location_f2 = self.prop_lt_f2.get_location()
        norm1 = self.compute_magnitude(current_location_f1, current_location_ego)
        norm2 = self.compute_magnitude(current_location_f2, current_location_ego)
        self.current_transform = self.prop_lt_ego.get_transform()
        self.yaw_vehicle = self.current_transform.rotation.yaw
        self.yaw_target_road = self.next_lane_target.transform.rotation.yaw
        diff_angle = abs(abs(int(179 + self.yaw_vehicle)) - abs(int(self.yaw_target_road)))
        v_ego = self.prop_lt_ego.get_velocity()
        v1 = self.prop_lt_f1.get_velocity()
        v2 = self.prop_lt_f2.get_velocity()
        kmh_ego = int(3.6 * math.sqrt(v_ego.x ** 2 + v_ego.y ** 2 + v_ego.z ** 2))
        kmh1 = int(3.6 * math.sqrt(v1.x ** 2 + v1.y ** 2 + v1.z ** 2))
        kmh2 = int(3.6 * math.sqrt(v2.x ** 2 + v2.y ** 2 + v2.z ** 2))
        diff_v = kmh1 - kmh2
        diff_v1 = kmh2 - kmh_ego
        self.x_speed_range = np.arange(0, 80, 5, np.float32)
        self.x_sd_range = np.arange(0, 20, 1, np.float32)
        self.y_powder_range = np.arange(0, 1, 0.1, np.float32)
        current_transform = self.prop_lt_ego.get_transform()
        self.yaw_vehicle = current_transform.rotation.yaw
        target_transform = self.prop_lt_ego.get_transform()
        self.yaw_target_road = target_transform.rotation.yaw
        diff_angle = abs(abs(int(179 + self.yaw_vehicle)) - abs(int(self.yaw_target_road)))
        x_speed = ctrl.Antecedent(self.x_speed_range, 'stain')
        x_sd = ctrl.Antecedent(self.x_sd_range, 'oil')
        y_powder = ctrl.Consequent(self.y_powder_range, 'powder')
        x_speed['N'] = fuzz.trimf(self.x_speed_range, [0, 0, 40])
        x_speed['M'] = fuzz.trimf(self.x_speed_range, [20, 40, 60])
        x_speed['P'] = fuzz.trimf(self.x_speed_range, [40, 80, 80])
        x_sd['N'] = fuzz.trimf(self.x_sd_range, [0, 0, 10])
        x_sd['M'] = fuzz.trimf(self.x_sd_range, [5, 10, 15])
        x_sd['P'] = fuzz.trimf(self.x_sd_range, [10, 20, 20])
        y_powder['N'] = fuzz.trimf(self.y_powder_range, [0, 0, 0.5])
        y_powder['M'] = fuzz.trimf(self.y_powder_range, [0.25, 0.5, 0.75])
        y_powder['P'] = fuzz.trimf(self.y_powder_range, [0.5, 1, 1])

        y_powder.defuzzify_method = 'centroid'
        rule0 = ctrl.Rule(antecedent=((x_speed['N'] & x_sd['N']) |
                                      (x_speed['N'] & x_sd['M'])),
                          consequent=y_powder['N'], label='rule N')
        rule1 = ctrl.Rule(antecedent=(
                (x_speed['M'] & x_sd['M']) |
                (x_speed['N'] & x_sd['P']) |
                (x_speed['M'] & x_sd['N'])),
            consequent=y_powder['M'], label='rule M')
        rule2 = ctrl.Rule(antecedent=((x_speed['P'] & x_sd['N']) |
                                      (x_speed['M'] & x_sd['P']) |
                                      (x_speed['P'] & x_sd['M']) |
                                      (x_speed['P'] & x_sd['P'])),
                          consequent=y_powder['P'], label='rule P')
        system = ctrl.ControlSystem(rules=[rule0, rule1, rule2])
        sim = ctrl.ControlSystemSimulation(system)
        sim.input['stain'] = kmh1
        sim.input['oil'] = diff_v1
        sim.compute()
        output_powder = sim.output['powder']
        if kmh_ego > 50:
            safe_dis = kmh_ego
        elif kmh_ego > 20:
            safe_dis = kmh_ego - 10
        else:
            safe_dis = 0.5 * kmh_ego

        if safe_dis - norm1 > 10:
            rewardsafe = 200
        elif safe_dis - norm1 == 0:
            rewardsafe = 0
        elif safe_dis - norm1 < 0:
            rewardsafe = -200
        else:
            rewardsafe = 100

        if diff_v > 10:
            rewardspeed = 200
        elif diff_v > 5:
            rewardspeed = 100
        elif diff_v == 0:
            rewardspeed = 0
        else:
            rewardspeed = -200
        if len(self.collision_hist) != 0:
            rewardrule = -1000
            done = True
        else:
            rewardrule = 0
        if diff_angle <= 5:
            rewardangle = 100
        else:
            rewardangle = -50
        reward = output_powder * rewardsafe + (1 - output_powder) * rewardspeed + rewardrule + rewardangle
        if self.episode_start + SECONDS_PER_EPISODE < time.time():
            done = True
        data = np.array(self.radar_data)
        return data[-LIMIT_RADAR:], [kmh_ego, self.yaw_vehicle, self.lane_id_target, diff_angle,
                                     norm1], reward, done, None


    def left_lane_change(self, action):
        done = False
        # if action == 0:
        #     self.prop_lt_ego.awpply_control(carla.VehicleControl(throttle=self.a3, steer=0.3))
        # else:
        #     self.prop_lt_ego.apply_control(carla.VehicleControl(throttle=self.a3, steer=0))
        action[0] = np.random.uniform(0.3, 0.1)
        action[1] = np.random.uniform(0.2, -0.1)
        self.prop_lt_ego.apply_control(
            carla.VehicleControl(throttle=action[0], steer=action[1], brake=action[2], reverse=action[3]))
        self.collision_list = []
        self.lane_id_ego = self.map.get_waypoint(self.prop_lt_ego.get_location()).lane_id
        self.lane_id_target = self.next_lane_target.lane_id
        current_location_ego = self.prop_lt_ego.get_location()
        current_location_f1 = self.prop_lt_f1.get_location()
        current_location_f2 = self.prop_lt_f2.get_location()
        # target_f1 = self.target_waypoint1.get_location()
        # target_f2 = self.target_waypoint2.get_location()
        norm1 = self.compute_magnitude(current_location_f1, current_location_ego)
        norm2 = self.compute_magnitude(current_location_f2, current_location_ego)
        # target_norm = self.compute_magnitude(target_waypoint1, current_location_ego)
        # target_norm1 = self.compute_magnitude(target_waypoint2, current_location_ego)
        self.current_transform = self.prop_lt_ego.get_transform()
        self.yaw_vehicle = self.current_transform.rotation.yaw
        self.yaw_target_road = self.next_lane_target.transform.rotation.yaw
        diff_angle = abs(abs(int(179 + self.yaw_vehicle)) - abs(int(self.yaw_target_road)))
        v_ego = self.prop_lt_ego.get_velocity()
        v1 = self.prop_lt_f1.get_velocity()
        v2 = self.prop_lt_f2.get_velocity()

        kmh_ego = int(3.6 * math.sqrt(v_ego.x ** 2 + v_ego.y ** 2 + v_ego.z ** 2))
        kmh1 = int(3.6 * math.sqrt(v1.x ** 2 + v1.y ** 2 + v1.z ** 2))
        kmh2 = int(3.6 * math.sqrt(v2.x ** 2 + v2.y ** 2 + v2.z ** 2))
        diff_v = kmh1 - kmh2
        diff_v1 = kmh2 - kmh_ego
        self.x_speed_range = np.arange(0, 80, 5, np.float32)
        self.x_sd_range = np.arange(0, 20, 1, np.float32)
        self.y_powder_range = np.arange(0, 1, 0.1, np.float32)
        current_transform = self.prop_lt_ego.get_transform()
        self.yaw_vehicle = current_transform.rotation.yaw
        target_transform = self.prop_lt_ego.get_transform()
        self.yaw_target_road = target_transform.rotation.yaw
        diff_angle = abs(abs(int(179 + self.yaw_vehicle)) - abs(int(self.yaw_target_road)))
        x_speed = ctrl.Antecedent(self.x_speed_range, 'stain')
        x_sd = ctrl.Antecedent(self.x_sd_range, 'oil')
        y_powder = ctrl.Consequent(self.y_powder_range, 'powder')
        x_speed['N'] = fuzz.trimf(self.x_speed_range, [0, 0, 40])
        x_speed['M'] = fuzz.trimf(self.x_speed_range, [20, 40, 60])
        x_speed['P'] = fuzz.trimf(self.x_speed_range, [40, 80, 80])
        x_sd['N'] = fuzz.trimf(self.x_sd_range, [0, 0, 10])
        x_sd['M'] = fuzz.trimf(self.x_sd_range, [5, 10, 15])
        x_sd['P'] = fuzz.trimf(self.x_sd_range, [10, 20, 20])
        y_powder['N'] = fuzz.trimf(self.y_powder_range, [0, 0, 0.5])
        y_powder['M'] = fuzz.trimf(self.y_powder_range, [0.25, 0.5, 0.75])
        y_powder['P'] = fuzz.trimf(self.y_powder_range, [0.5, 1, 1])

        y_powder.defuzzify_method = 'centroid'
        rule0 = ctrl.Rule(antecedent=((x_speed['N'] & x_sd['N']) |
                                      (x_speed['N'] & x_sd['M'])),
                          consequent=y_powder['N'], label='rule N')
        rule1 = ctrl.Rule(antecedent=(
                (x_speed['M'] & x_sd['M']) |
                (x_speed['N'] & x_sd['P']) |
                (x_speed['M'] & x_sd['N'])),
            consequent=y_powder['M'], label='rule M')
        rule2 = ctrl.Rule(antecedent=((x_speed['P'] & x_sd['N']) |
                                      (x_speed['M'] & x_sd['P']) |
                                      (x_speed['P'] & x_sd['M']) |
                                      (x_speed['P'] & x_sd['P'])),
                          consequent=y_powder['P'], label='rule P')
        system = ctrl.ControlSystem(rules=[rule0, rule1, rule2])
        sim = ctrl.ControlSystemSimulation(system)
        sim.input['stain'] = kmh1
        sim.input['oil'] = diff_v1
        sim.compute()
        output_powder = sim.output['powder']
        if kmh_ego > 50:
            safe_dis = kmh_ego
        elif kmh_ego > 20:
            safe_dis = kmh_ego - 10
        else:
            safe_dis = 0.5 * kmh_ego

        if safe_dis - norm1 > 10:
            rewardsafe = 200
        elif safe_dis - norm1 == 0:
            rewardsafe = 0
        elif safe_dis - norm1 < 0:
            rewardsafe = -200
        else:
            rewardsafe = 100

        if diff_v >10:
            rewardspeed = 200
        elif diff_v >5:
            rewardspeed = 100
        elif diff_v == 0:
            rewardspeed = 0
        else:
            rewardspeed = -200
        if len(self.collision_hist) != 0:
            rewardrule = -1000
            done = True
        else:
            rewardrule = 0
        if diff_angle <= 5:
            rewardangle = 100
        else:
            rewardangle = -50
        reward = output_powder * rewardsafe + (1-output_powder)* rewardspeed + rewardrule + rewardangle


        # reward = output_powder * (safe_dis - norm1)+(1 - output_powder) * diff_v/2
        # if self.lane_id_ego != self.lane_id_target:
        #     if (action[1] < 0):
        #         reward = reward - 20
        #     elif (action[1] > 0):
        #         if (action[1] < 0.1):
        #             reward = reward + 100
        #         else:
        #             reward = reward + 10
        #     else:
        #         reward = reward + 20
        #     if len(self.collision_hist) != 0:
        #         reward = reward - 1000
        #         done = True
        # else:
        #     reward = reward + 50
        #     if (diff_angle > 5):
        #         if (action[1] < 0):
        #             reward = reward - 10
        #         if (action[1] > 0):
        #             reward = reward - 30
        #     if len(self.collision_hist) != 0:
        #         reward = reward - 1000
        #         done = True
        #     if (diff_angle <= 5):
        #         done = True
        #         reward = reward + 50
        if self.episode_start + SECONDS_PER_EPISODE < time.time():
            done = True
        data = np.array(self.radar_data)
        return data[-LIMIT_RADAR:], [kmh_ego, self.yaw_vehicle, self.lane_id_target, diff_angle,
                                     norm1], reward, done, None

    def compute_magnitude_angle(self, target_location, current_location, target_yaw, current_yaw):
        target_vector = np.array([target_location.x - current_location.x, target_location.y - current_location.y])
        norm_target = np.linalg.norm(target_vector)
        current_yaw_rad = abs(current_yaw)
        target_yaw_rad = abs(target_yaw)
        diff = current_yaw_rad - target_yaw_rad
        diff_angle = (abs((diff)) % 180.0)
        return (norm_target, diff_angle)

    def compute_magnitude(self, target_location, current_location):
        target_vector = np.array([target_location.x - current_location.x, target_location.y - current_location.y])
        norm_target = np.linalg.norm(target_vector)
        return norm_target

# if __name__ == "__main__":
#     CarlaVehicle()
