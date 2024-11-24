from multiprocessing import Process,Queue
from traci_sim import run_controllable_sumo


if __name__ == "__main__":
    q1 = Queue()

    csumo = Process(target=run_controllable_sumo, args=(q1,))
    csumo.start()

    while True:
        msg = input()

        if msg == "q":
            csumo.close()
            exit(0)
        q1.put(msg)