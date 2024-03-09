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
light_int = 0
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
    log_config = LogConfig(name='Light values', period_in_ms=50)
    log_config.add_variable('stateEstimate.x', 'float')
    log_config.add_variable('stateEstimate.y', 'float')
    log_config.add_variable('stateEstimate.yaw', 'float')

    log_config.add_variable('synthLog.q1dist', 'uint8_t')
    log_config.add_variable('synthLog.q2dist', 'uint8_t')
    log_config.add_variable('synthLog.q3dist', 'uint8_t')
    log_config.add_variable('synthLog.q4dist', 'uint8_t')
    log_config.add_variable['synthLog.light_intensity']
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
    cflib.crtp.init_drivers()
    i = 0
    
    epsilon = 12 
    sigma_const = 0.6
    alpha = 1.0
    beta = 2.0
    u_max = 0.08
    w_max = 1.5708/2.5
    constant = 325 * (0.021 / 0.1)
    K1 = 0.2
    K2 = 0.1

    with ClientAsync() as client:
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
            try:
                while True:
                    if time.time() - time_last >= 0.05:

                        # bearings = np.array([1.571/2, 1.571/2 + 1.571, 1.571/2 + 1.571*2, 1.571/2 + 1.571*3]) 
                        bearings = np.array([0.0, 1.5708, np.pi, -1.5708]) 
                        k = 4
                        distances = np.array([q1dist, q2dist, q3dist, q4dist])
                        distances[distances > 1.99] = np.inf
                        headings = np.array([q1h, q2h, q3h, q4h])
                        own_heading = pos_hs

                        pi_s = epsilon * (2 * (np.divide(np.power(sigma_const, 4), np.power(distances, 5))) - (
                        np.divide(np.power(sigma_const, 2), np.power(distances, 3))))
                        px_s = np.multiply(pi_s, np.cos(np.array(bearings)))
                        py_s = np.multiply(pi_s, np.sin(np.array(bearings)))
                        pbar_xs = np.nansum(px_s, axis=0)
                        pbar_ys = np.nansum(py_s, axis=0)

                        if len(headings[distances < 1.99]) > 0:
                            angle_av = np.mean(headings[distances < 1.99]) - pos_hs
                        else:
                            angle_av = pos_hs

                        hbar_x = np.cos(angle_av)
                        hbar_y = np.sin(angle_av)

                        f_x = alpha * pbar_xs + beta * hbar_x
                        f_y = alpha * pbar_ys + beta * hbar_y
                        f_mag = np.sqrt(np.square(f_x) + np.square(f_y))
                        glob_ang = np.arctan2(f_y, f_x)
                        u = K1 * np.multiply(f_mag, np.cos(glob_ang)) + 0.05

                        if u > u_max:
                            u = u_max
                        elif u < 0:
                            u = 0.0
                        w = K2 * np.multiply(f_mag, np.sin(glob_ang))
                        if w > w_max:
                            w = w_max
                        elif w < -w_max:
                            w = -w_max

                        print(f"u: {u}, w: {w}, X: {pos_xs}, Y: {pos_ys}, h: {pos_hs}, li: {light_int}, distances: {distances},")

                        left = constant * (u - (w*2.75 / 2) * 0.085) / 0.021
                        right = constant * (u + (w*2.75 / 2) * 0.085) / 0.021
                        if np.isnan([u, w]).any():
                            left = 0.0
                            right = 0.0
                            raise ValueError

                        
                    else:
                        u = 0.0
                        w = 0.0
                        left = 0
                        right = 0


                    time_last = time.time()
                    i += 1

                    targets_g = {"motor.left.target": [int(left)], "motor.right.target": [int(right)]}
                    call_program()
            except Exception as e:
                print("Terminated:")
                print(e.with_traceback())
                # np.save('./logs/log_quad_dist.npy', log_quadrant_distance)
                # np.save('./logs/log_neg_headings.npy', log_neg_rel_heading)
                os.system("python3 -m tdmclient run --stop")
                print("exiting program")
                sys.exit()
                
