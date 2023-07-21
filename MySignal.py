import signal
import time

class MySignal(object):
    """Signal processing test"""
    bFinish = False

    def __init__(self) -> None:
        signal.signal(signal.SIGINT, MySignal.handler)

    def handler(signum, frame):
        """Signal processing handler"""
        signame = signal.Signals(signum).name
        print(f'Signal handler called with signal {signame} ({signum})')
        MySignal.bFinish = True

    def run(self, timeout=20):
        count = 0
        while count<timeout:
            print(f"{count}", end="\r", flush=True)
            if MySignal.bFinish:
                print("Finish flag detected")
                break
            time.sleep(1)
            count += 1

if __name__ == '__main__':
    ms = MySignal()
    ms.run()