import multiprocessing
import sys
import traceback
import tellopy
import av
import cv2
import numpy as np
import time
from simple_pid import PID
import queue
from time import sleep
from ctypes import c_bool
import matplotlib.pyplot as plt

def detect_marker_cord(frame):
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    corners, ids, _ = detector.detectMarkers(frame)
    if ids is not None and len(corners) > 0:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        return corners
    return None

def find_error_x(corners):
    pts = np.squeeze(corners[0])
    img_point1=np.array(pts,dtype=np.float32)
    if img_point1.ndim != 2 or img_point1.shape[0] != 4:
        return 0
    center = img_point1.mean(axis=0)
    desired = 480
    return desired - center[0]

def find_error_y(corners):
    pts = np.squeeze(corners[0])
    img_point1=np.array(pts,dtype=np.float32)
    if img_point1.ndim != 2 or img_point1.shape[0] != 4:
        return 0
    center = img_point1.mean(axis=0)
    desired = 360
    return desired - center[1]

def find_error_z(corners):
    pts = np.squeeze(corners[0])
    img_point1=np.array(pts,dtype=np.float32)
    if img_point1.ndim != 2 or img_point1.shape[0] != 4:
        return 0
    diag = np.linalg.norm(pts[1] - pts[3])
    desired = 100
    return desired - diag

class Video:
    def __init__(self):
        self.q = multiprocessing.Queue()
        self.run = multiprocessing.Value(c_bool, True)

    @staticmethod
    def clean_queue(q: multiprocessing.Queue):
        while not q.empty():
            try:
                q.get_nowait()
            except queue.Empty:
                break

    def video_get(self):

        pid_x = PID(0.0014, 0,0.0005, setpoint=0); pid_x.output_limits = (-1,1)
        pid_y = PID(0.0014, 0,0.00005, setpoint=0); pid_y.output_limits = (-1,1)
        pid_z = PID(0.003,    0.00000000277, 0.0000007, setpoint=0); pid_z.output_limits = (-1,1)


        drone = tellopy.Tello()
        try:
            drone.connect()
            drone.wait_for_connection(60)
            container = av.open(drone.get_video_stream())
            drone.takeoff()
            #drone.land()

            for frame in container.decode(video=0):
                if not self.run.value:
                    drone.land()
                    self.run.value = True

                img = np.array(frame.to_image())
                corners = detect_marker_cord(img)


                if corners is not None:
                    ex = find_error_x(corners)
                    ey = find_error_y(corners)
                    ez = find_error_z(corners)
                else:
                    ex = ey = ez = 0

                cmd_x = pid_x(ex)
                cmd_y = -pid_y(ey)
                cmd_z = -pid_z(ez)


                drone.set_roll   (cmd_x)
                drone.set_throttle(cmd_y)
                drone.set_pitch  (cmd_z)

                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                self.q.put((gray, cmd_x, cmd_y, cmd_z))
                Video.clean_queue(self.q)

        except Exception:
            traceback.print_exc(file=sys.stderr)
        finally:
            try:
                drone.land()
                drone.quit()
            except:
                pass

    def start(self):
        multiprocessing.Process(target=self.video_get, daemon=True).start()


def main():

    vr = Video()
    vr.start()


    plt.ion()
    fig, ax = plt.subplots(figsize=(7, 4))
    time_data = []
    x_data = []
    y_data = []
    z_data = []
    t0 = time.time()

    while True:
        try:
            frame, vx, vy, vz = vr.q.get(timeout=5)
        except queue.Empty:
            continue


        cv2.imshow('Drone Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        elif cv2.waitKey(1) & 0xFF == ord('l'):
            vr.run.value = False


        t = time.time() - t0
        time_data.append(t)
        x_data.append(vx)
        y_data.append(vy)
        z_data.append(vz)


        ax.clear()
        ax.plot(time_data, x_data, label='ПИД X')
        ax.plot(time_data, y_data, label='PID Y')
        ax.plot(time_data, z_data, label='PID Z')
        ax.set_xlabel('время,с')
        ax.set_ylabel('Выход')
        ax.set_title('Continuous PID Outputs (X, Y, Z)')
        ax.legend()
        ax.relim()
        ax.autoscale_view()

        fig.canvas.draw()
        fig.canvas.flush_events()
        fig.savefig('pidout.svg', format='svg')


    cv2.destroyAllWindows()
    plt.ioff()
    plt.show()


if __name__ == '__main__':
    main()

