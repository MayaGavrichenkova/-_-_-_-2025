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
            total_time = 40
            meander_started = False
            t0 = None
            amplitude = 0.3  # вертикальное смещение
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
                    # начинаем вертикальный меандр
                    drone.set_throttle(amplitude)
                    print("Меандр начат в", t0)
                    time.sleep(2)
                    drone.set_throttle(-amplitude)
                    continue

                meander_val = 0.0
                if meander_started and t0 is not None:
                    elapsed = time.time() - t0
                    if elapsed >= total_time:
                        drone.land()
                        print("Посадка в", time.time())
                        break
                    phase = (elapsed % period) < half_period
                    meander_val = amplitude if phase else -amplitude
                    drone.set_throttle(meander_val)

                # добавляем в очередь данные
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

    CSV_NAME = 'drone_coord.csv'
    with open(CSV_NAME, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['t_rel (s)', 'dx', 'dy', 'vertical_meander'])

    times, xs, ys = [], [], []
    t_meander, vals_meander = [], []

    plt.ion()
    fig_x, ax_x = plt.subplots(figsize=(7, 4))
    fig_y, ax_y = plt.subplots(figsize=(7, 4))
    fig_m, ax_m = plt.subplots(figsize=(7, 4))

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
        x_m = abs(dx) * PIXEL_TO_M
        y_m = abs(dy) * PIXEL_TO_M

        with open(CSV_NAME, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([f"{t_rel:.3f}", f"{x_m:.3f}", f"{y_m:.3f}", f"{meander_val:.3f}"])

        times.append(t_rel)
        xs.append(x_m)
        ys.append(y_m)
        t_meander.append(t_rel)
        vals_meander.append(meander_val)

        # смещение по X
        ax_x.clear()
        ax_x.plot(times, xs, label='dx, м')
        ax_x.set_xlabel('Время, с')
        ax_x.set_ylabel('Смещение X, м')
        ax_x.set_title('Переходный процесс по X')
        ax_x.legend()
        fig_x.tight_layout(); fig_x.canvas.draw(); fig_x.canvas.flush_events(); fig_x.savefig('step_response_dx.svg')

        # смещение по Y
        ax_y.clear()
        ax_y.plot(times, ys, label='dy, м')
        ax_y.set_xlabel('Время, с')
        ax_y.set_ylabel('Смещение Y, м')
        ax_y.set_title('Переходный процесс по Y')
        ax_y.legend()
        fig_y.tight_layout(); fig_y.canvas.draw(); fig_y.canvas.flush_events(); fig_y.savefig('step_response_dy.svg')

        # вертикальный меандр
        ax_m.clear()
        ax_m.plot(t_meander, vals_meander, label='Вертикальный меандр ±0.3, полупериод 2 с')
        ax_m.set_xlabel('Время, с')
        ax_m.set_ylabel('Амплитуда')
        ax_m.set_title('Колебания вверх-вниз')
        ax_m.set_ylim(-0.4, 0.4)
        ax_m.legend()
        fig_m.tight_layout(); fig_m.canvas.draw(); fig_m.canvas.flush_events(); fig_m.savefig('vertical_meander.svg')

    cv2.destroyAllWindows(); plt.ioff()


if __name__ == '__main__':
    main()
