import math
import time
import sys
import datetime


class SimpleLogger:
    def __init__(self):
        self.starting_time = time.time()
        self.dict = {}
        self.its = 0  # how many iterations, or epochs
        with open('./logging.log', 'a') as f:
            f.write(' ------------------------ \n')

    @staticmethod
    def print(x: str='', end: str='\n'):
        print(x, end=end)
        with open('./logging.log', 'a') as f:
            f.write(x + end)

    def report(self, cur_it, max_it):
        if self.its == 0:
            print('no records')
            return

        ending_time = time.time()
        delta = ending_time - self.starting_time
        self.starting_time = ending_time

        now = datetime.datetime.now()
        now_string = f'{now.date()} {now.hour}:{now.minute}:{now.second}'
        self.print(f'[{cur_it:06d}/{max_it} | time elapsed: {int(delta)}s | {now_string}]')

        for k, v in self.dict.items():
            self.print(f'{k}: {v / self.its:.06f}', end='   ')
            self.dict[k] = 0
        self.print()
        self.its = 0
        sys.stdout.flush()

    # noinspection PyUnusedLocal
    def log(self, cur_it, max_it, **kwargs):
        for k, v in kwargs.items():
            if math.isnan(v):
                raise RuntimeError(f"One `NaN` value is encountered: {k}")
            if k not in self.dict:
                self.dict[k] = 0.0
            self.dict[k] += v
        self.its += 1


def test():
    lg = SimpleLogger()

    lg.log(0, 10, a=100, b=200)
    lg.log(1, 10, a=120, b=210)
    lg.log(2, 10, a=110, b=230)
    lg.report(2, 10)

    import time
    time.sleep(5)
    lg.log(3, 10, a=110, b=230)
    lg.log(4, 10, a=110, b=230)
    lg.log(5, 10, a=110, b=230)
    lg.report(5, 10)

    print('reading ./logging.log ... ')
    with open('./logging.log', 'r') as f:
        print(f.readlines())
    import os
    os.remove('./logging.log')


if __name__ == '__main__':
    test()
