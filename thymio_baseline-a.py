import os
import sys
import signal
import time
import threading
import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.crazyflie.syncLogger import SyncLogger
import numpy as np
from tdmclient import ClientAsync
from Controllers.ControlThymio import NNController, adaptiveNNController

uri = 'usb://0'
global pos_xs, pos_ys, pos_hs, q1dist, q2dist, q3dist, q4dist, q1h, q2h, q3h, q4h, light_int, log_quadrant_distance, log_neg_rel_heading
pos_xs = np.array([])
pos_ys = np.array([])
log_quadrant_distance = np.array([])
log_neg_rel_heading = np.array([])
pos_hs = 0.0
q1dist = 99
q2dist = 99
q3dist = 99
q4dist = 99
q1h = 0
q2h = 0
q3h = 0
q4h = 0

# Initial values for u and w
u = 0
w = 0
start_time = 0


def activate_high_level_commander(cf):
    cf.param.set_value('commander.enHighLevel', '1')


def reset_estimator(cf):
    cf.param.set_value('kalman.resetEstimation', '1')
    time.sleep(0.1)
    cf.param.set_value('kalman.resetEstimation', '0')
    time.sleep(1.0)
    cf.param.set_value('fmodes.if_takeoff', '1')


def log_cf(scf):
    global pos_xs, pos_ys, pos_hs, q1dist, q2dist, q3dist, q4dist, q1h, q2h, q3h, q4h, light_int, log_quadrant_distance, log_neg_rel_heading
    log_config = LogConfig(name='Light values', period_in_ms=100)
    log_config.add_variable('stateEstimate.x', 'float')
    log_config.add_variable('stateEstimate.y', 'float')
    log_config.add_variable('stateEstimate.yaw', 'float')

    log_config.add_variable('synthLog.q1dist', 'uint8_t')
    log_config.add_variable('synthLog.q2dist', 'uint8_t')
    log_config.add_variable('synthLog.q3dist', 'uint8_t')
    log_config.add_variable('synthLog.q4dist', 'uint8_t')
    log_config.add_variable('synthLog.light_intensity')
    #
    log_config.add_variable('synthLog.q1h', 'uint8_t')
    log_config.add_variable('synthLog.q2h', 'uint8_t')
    log_config.add_variable('synthLog.q3h', 'uint8_t')
    log_config.add_variable('synthLog.q4h', 'uint8_t')

    with SyncLogger(scf, log_config) as logger:
        for log_entry in logger:
            data = log_entry[1]

            pos_xs = data['stateEstimate.x']
            pos_ys = data['stateEstimate.y']
            pos_hs = data['stateEstimate.yaw'] * 0.0174532
            q1dist = data['synthLog.q1dist'] * 2 / 255
            q2dist = data['synthLog.q2dist'] * 2 / 255
            q3dist = data['synthLog.q3dist'] * 2 / 255
            q4dist = data['synthLog.q4dist'] * 2 / 255
            light_int = data['synthLog.light_intensity']
            log_quadrant_distance = np.append(log_quadrant_distance, q4dist)
            q1h = data['synthLog.q1h'] / (255 / (3.141592*2))
            q2h = data['synthLog.q2h'] / (255 / (3.141592*2))
            q3h = data['synthLog.q3h'] / (255 / (3.141592*2))
            q4h = data['synthLog.q4h'] / (255 / (3.141592*2))
            log_neg_rel_heading = np.append(log_neg_rel_heading, q4h)


if __name__ == '__main__':
    ip_end = os.popen('ifconfig | grep -oE "(inet 10.15.3.[0-9]{1,3})"').read()[:-1].split('.')[-1]

    adaptive = True
    n_subs = 2

    cflib.crtp.init_drivers()
    i = 0

    u_max = 0.08
    w_max = 1.5708/2.5
    constant = 325 * (0.021 / 0.1)

    experiment_folder = "./results/sim/Baseline-a/4"
    reservoir_dirs = [experiment_folder + f'/subgroup_{n_sub}' for n_sub in range(n_subs)]
    x_best = np.load(f"{experiment_folder}/x_best.npy")[-1]

    if adaptive:
        controller = adaptiveNNController(9,2)
        controller.load_geno(reservoir_dirs)
        ## Load Controller
        genotype = [x_best[n_sub * 18:(1 + n_sub) * 18] for n_sub in range(n_subs)]

        controller.geno2pheno(genotype)
    else:
        controller = NNController(9, 2)
        n_sub = int(ip_end <= '86')

        controller.load_geno(reservoir_dirs[n_sub])
        rgb = [n_sub , 1 - n_sub, 0]
        genotype = x_best[99][n_sub * 18:(1 + n_sub) * 18]
        controller.geno2pheno(genotype)


    with ClientAsync() as client:
        client.DEFAULT_SLEEP /= 10
        async def change_node_var():
            with await client.lock() as node:
                await node.watch(variables=True)
                await node.set_variables(targets_g)
                # node.add_variables_changed_listener(on_variables_changed)


        def call_program():
            client.run_async_program(change_node_var)


        with SyncCrazyflie(uri, cf=Crazyflie(rw_cache='./cache')) as scf:
            cf = scf.cf
            thread_1 = threading.Thread(target=log_cf, args=([scf]))
            thread_1.start()
            reset_estimator(cf)
            draw_angle = 1.5708/2

            time_last = time.time()
            u = 0.0
            w = 0.0
            left = 0
            right = 0

            init_rot = np.random.rand()*2*np.pi - np.pi
            left = constant * (- (init_rot / 2) * 0.085) / 0.021 * 0.2
            right = constant * ((init_rot / 2) * 0.085) / 0.021 * 0.2
            targets_g = {"motor.left.target": [int(left)], "motor.right.target": [int(right)]}
            while time.time() - time_last < 5:
                call_program()
            targets_g = {"motor.left.target": [int(0)], "motor.right.target": [int(0)]}
            call_program()
            waiter = input("Enter new values for u and w, separated by a space: ")
            time_last = time.time()
            try:
                while True:
                    if time.time() - time_last >= 0.05:
                        # print(time.time() - time_last)

                        # bearings = np.array([1.571/2, 1.571/2 + 1.571, 1.571/2 + 1.571*2, 1.571/2 + 1.571*3])
                        bearings = np.array([0.0, 1.5708, np.pi, -1.5708])
                        k = 4
                        distances = np.array([q1dist, q2dist, q3dist, q4dist])
                        headings = np.array([q1h, q2h, q3h, q4h])
                        own_heading = np.array(pos_hs)
                        headings_rel = (headings-own_heading+np.pi)%(2*np.pi)-np.pi
                        headings_rel[distances==2.0] = 0
                        # distances[distances!=2.0] = np.sqrt(distances[distances!=2.0]-1.0)*2
                        # distances[distances!=2.0] = distances[distances!=2.0]
                        distances[distances==2.0] = 2.01
                        distances[distances<0] = 0



                        if own_heading < 0:
                            own_heading = own_heading + (3.141592*2)

                        # Get velocity commands
                        state = np.hstack((distances,
                                           headings_rel,
                                           light_int))
                        u, w = controller.velocity_commands(state)

                        print(f"u: {u}, w: {w}, X: {pos_xs}, Y: {pos_ys}, li: {light_int},\nh: {headings_rel}\ndistances: {distances},")

                        # left = constant * (u - (w * 2.75 / 2) * 0.085) / 0.021
                        # right = constant * (u + (w * 2.75 / 2) * 0.085) / 0.021

                        left = constant * (u + 0.025 - (w / 2) * 0.085) / 0.021 / 1.5
                        right = constant * (u + 0.025 + (w / 2) * 0.085) / 0.021 / 1.5
                        if np.isnan([u, w]).any():
                            left = 0.0
                            right = 0.0
                            raise ValueError
                        time_last = time.time()
                        # else:
                        #     u = 0.0
                        #     w = 0.0
                        #     left = 0
                        #     right = 0
                        i += 1
                        targets_g = {"motor.left.target": [int(left)], "motor.right.target": [int(right)]}
                        call_program()
            except KeyboardInterrupt as e:
                print("KeyBoard interrupt detected!")
                # np.save('./logs/log_quad_dist.npy', log_quadrant_distance)
                # np.save('./logs/log_neg_headings.npy', log_neg_rel_heading)
                os.system("python3 -m tdmclient run --stop")
                print("exiting program")
                sys.exit()
            except Exception as e:
                print("Terminated:")
                print(e.with_traceback())
                # np.save('./logs/log_quad_dist.npy', log_quadrant_distance)
                # np.save('./logs/log_neg_headings.npy', log_neg_rel_heading)
                os.system("python3 -m tdmclient run --stop")
                print("exiting program")
                sys.exit()
