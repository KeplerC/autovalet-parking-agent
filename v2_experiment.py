from v2_experiment_utils import (
    load_client,
    is_done,
    town04_load,
    town04_spectator_bev,
    town04_spawn_ego_vehicle,
    town04_spawn_parked_cars,
    town04_spectator_follow,
    town04_get_drivable_graph,
    town04_spawn_moving_cars
)

# For lane waypoint hack
from parking_position import (
    parking_lane_waypoints_Town04
)

import numpy as np
import matplotlib.pyplot as plt
import imageio.v3 as iio

SCENARIOS = [
    (17, [16, 18]),
    (18, [17, 19]),
    (19, [18, 20]),
    (20, [19, 21]),
    (21, [20, 22]),
    (22, [21, 23]),
    (23, [22, 24]),
    (24, [23, 25]),
    (25, [24, 26]),
    (26, [25, 27]),
    (27, [26, 28]),
    (28, [27, 29]),
    (29, [28, 30]),
    (30, [29, 31]),
    (31, [30, 32]),
    (32, [31, 33]),
    (33, [32, 34]),
    (34, [33, 35]),
    (35, [34, 36]),
    (36, [35, 37]),
    (37, [36, 38]),
    (38, [37, 39]),
    (39, [38, 40]),
    (40, [39, 41]),
    (41, [40, 42]),
    (42, [41, 43]),
    (43, [42, 44]),
    (44, [43, 45]),
    (45, [44, 46]),
    (46, [45, 47]),
    (47, [46, 48]),
]
NUM_RANDOM_CARS = 2

def run_scenario(world, destination_parking_spot, parked_spots, ious, recording_file):
    try:
        # load parked cars
        parked_cars, parked_cars_bbs = town04_spawn_parked_cars(world, parked_spots, destination_parking_spot, NUM_RANDOM_CARS)
        
        # spawn moving cars
        moving_cars = town04_spawn_moving_cars(world, num_cars=1, target_speed=5)

        # load car
        car = town04_spawn_ego_vehicle(world, destination_parking_spot)
        recording_cam = car.init_recording(recording_file)

        # HACK: enable perfect perception of parked cars
        car.car.obs = parked_cars_bbs

        # HACK: set lane waypoints to guide parking in adjacent lanes
        car.car.lane_wps = parking_lane_waypoints_Town04

        # run simulation
        i = 0
        while not is_done(car):
            # Update moving cars
            for moving_car, agent in moving_cars:
                control = agent.run_step()
                if control:
                    control.steering = 0.0
                    control.steer = 0.0
                    control.brake = 0.0
                    print(f'control: {control}')
                    
                    moving_car.apply_control(control)
            
            # Update dynamic obstacles for the ego vehicle
            dynamic_obs = []
            for moving_car, _ in moving_cars:
                loc = moving_car.get_location()
                bb = moving_car.bounding_box
                dynamic_obs.append([
                    loc.x - bb.extent.x, loc.y - bb.extent.y,
                    loc.x + bb.extent.x, loc.y + bb.extent.y
                ])
            car.car.obs = parked_cars_bbs + dynamic_obs
            
            # Check for collision
            if car.has_collided:
                print("Trial stopped early due to collision")
                break

            world.tick()
            car.run_step()
            car.process_recording_frames()

        # Only record IOU if no collision occurred
        if not car.has_collided:
            iou = car.iou()
            ious.append(iou)
            print(f'IOU: {iou}')
        else:
            print('Trial failed due to collision - no IOU recorded')

    finally:
        recording_cam.destroy()
        car.destroy()
        for parked_car in parked_cars:
            parked_car.destroy()
        for moving_car, agent in moving_cars:
            moving_car.destroy()

def main():
    try:
        client = load_client()

        # load map
        world = town04_load(client)

        # load spectator
        town04_spectator_bev(world)

        # load recording file
        recording_file = iio.imopen('./test.mp4', 'w', plugin='pyav')
        recording_file.init_video_stream('vp9', fps=5)

        # run scenarios
        ious = []
        num_scenarios = 2
        for destination_parking_spot, parked_spots in SCENARIOS:
            print(f'Running scenario: destination={destination_parking_spot}, parked_spots={parked_spots}')
            run_scenario(world, destination_parking_spot, parked_spots, ious, recording_file)
            num_scenarios -= 1
            if num_scenarios == 0:
                break

        # graph ious
        plt.clf()
        plt.boxplot(ious, positions=[1], vert=True, patch_artist=True, widths=0.5,
            boxprops=dict(facecolor='lightblue', color='blue'),
            medianprops=dict(color='red'))
        x_scatter = np.random.normal(loc=0.5, scale=0.05, size=len(ious))
        plt.scatter(x_scatter, ious, color='darkblue', alpha=0.6, label='Data Points')
        plt.xticks([1], ['IOU Values'])  # Set x-ticks at the boxplot
        plt.title('Parking IOU Values')
        plt.ylabel('IOU Value')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend()
        plt.savefig('iou_boxplot.png')

    except KeyboardInterrupt:
        print('stopping simulation')
    
    finally:
        recording_file.close()
        world.tick()

if __name__ == '__main__':
    main()