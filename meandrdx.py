import multiprocessing
import sys
import traceback
import tellopy
import av
import cv2
import numpy as np
import time
import queue
from ctypes import c_bool
import matplotlib.pyplot as plt
import csv


def detect_drone_position(frame):
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    corners, ids, _ = detector.detectMarkers(frame)
    if ids is not None and len(corners) > 0:
        pts = np.squeeze(corners[0])
        center = np.array(pts, dtype=np.float32).mean(axis=0)
        dx = center[0] - 480
        dy = center[1] - 360
        return dx, dy, frame
    return None, None, frame

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
        drone = tellopy.Tello()
        try:
            drone.connect()
            drone.wait_for_connection(60)
            drone.start_video()
            container = av.open(drone.get_video_stream())

            drone.takeoff()
            all_time = 40
            meander_started = False
            t0 = None
            amplitude = 0.3
            period = 4.0
            half_period = period / 2

            for frame in container.decode(video=0):
                if not self.run.value:
                    drone.land()
                    self.run.value = True

                img = np.array(frame.to_image())
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                small = cv2.resize(gray, (480, 360))
                dx, dy, proc_frame = detect_drone_position(small)

                if not meander_started and dx is not None:
                    meander_started = True
                    t0 = time.time()
                    drone.set_roll(amplitude)
                    print("Ступень подана в", t0)
                    time.sleep(2)
                    drone.set_roll(-amplitude)
                    continue

                meander_val = 0.0
                if meander_started and t0 is not None:
                    elapsed = time.time() - t0
                    if elapsed >= all_time:
                        drone.land()
                        print("Посадка в", time.time())
                        break
                    phase = (elapsed % period) < half_period
                    meander_val = amplitude if phase else -amplitude
                    drone.set_roll(meander_val)

               
                self.q.put((proc_frame, t0, dx, dy, meander_val))
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

PIXEL_TO_M = 0.00026


def main():
    video = Video()
    video.start()

    CSV_NAME = 'drone_coorddx.csv'
    with open(CSV_NAME, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['t_rel (s)', 'dx', 'dy', 'meander'])

    times, xs, ys = [], [], []
    meander_times, meander_vals = [], []

    plt.ion()
    fig1, ax1 = plt.subplots(figsize=(7, 4))
    fig2, ax2 = plt.subplots(figsize=(7, 4))

    amplitude = 0.3
    period = 4.0
    half_period = period / 2

    while True:
        try:
            frame, t0, dx, dy, meander_val = video.q.get(timeout=1)
        except queue.Empty:
            continue

        cv2.imshow('Видео дрона', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('l'):
            video.run.value = False

        if t0 is None or dx is None:
            continue

        t_rel = time.time() - t0
        dx_m = abs(dx) * PIXEL_TO_M
        dy_m = abs(dy) * PIXEL_TO_M

        with open(CSV_NAME, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([f"{t_rel:.3f}", f"{dx_m:.2f}", f"{dy_m:.2f}", f"{meander_val:.3f}"])

        times.append(t_rel)
        xs.append(dx_m)
        meander_times.append(t_rel)
        meander_vals.append(meander_val)

        ax1.clear()
        ax1.plot(times, xs, label='Смещение по X, м')
        ax1.set_xlabel('Время, с')
        ax1.set_ylabel('Смещение, м')
        ax1.set_title('Переходный процесс по X')
        ax1.legend()
        fig1.tight_layout()
        fig1.canvas.draw()
        fig1.canvas.flush_events()
        fig1.savefig('step_response_dx.svg')

  
        ax2.clear()
        ax2.plot(meander_times, meander_vals, label='Меандр ±0.3, полупериод 2 с')
        ax2.set_xlabel('Время, с')
        ax2.set_ylabel('Амплитуда')
        ax2.set_title('Генерация меандра')
        ax2.set_ylim(-0.4, 0.4)
        ax2.legend()
        fig2.tight_layout()
        fig2.canvas.draw()
        fig2.canvas.flush_events()
        fig2.savefig('meander.svg')

    cv2.destroyAllWindows()
    plt.ioff()


if __name__ == '__main__':
    main()


