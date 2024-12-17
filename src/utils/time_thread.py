import threading
import tqdm
import time

class TimeThread(threading.Thread):

    def __init__(self, time_limit_in_sec,
                 position=0,
                 disable=False,
                 leave=False):

        super(TimeThread, self).__init__()

        assert time_limit_in_sec > 0
        self.time_limit_in_sec = time_limit_in_sec
        self.end_time = None
        self.sleep_interval_in_sec = 1
        self.disable = disable
        self.leave = leave
        self.position = position

    def run(self):

        self.end_time = time.time() + self.time_limit_in_sec
        progress_bar = tqdm.tqdm(total=self.time_limit_in_sec,
                                 position=self.position,
                                 disable=self.disable,
                                 leave=self.leave)

        while time.time() < self.end_time:

            progress_bar.update(self.sleep_interval_in_sec)
            time.sleep(self.sleep_interval_in_sec)

        progress_bar.close()

    def stop(self):

        self.end_time = float("-inf")


if __name__ == "__main__":

    t1 = TimeThread(30, position=0)
    t2 = TimeThread(30, position=1)

    t1.start()
    t2.start()

    time.sleep(10)

    t1.stop()
    t2.stop()

    t1.join()
    t2.join()