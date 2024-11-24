import traci
import json
import time
import threading
import random
from multiprocessing import Queue

def run_simulation_bg():
    while True:
        traci.simulationStep()

def clean_up(prev_cars, expected_cars):
    print(f"Expected cars: {expected_cars}")
    for c in prev_cars:
        if c not in expected_cars:
            # The car can resume going since it's out of the camera viewport
            try:
                traci.vehicle.replaceStop(c, 0, "")
                # traci.vehicle.resume(c)
            except Exception as e:
                pass

def run(cmqQ = None):
    with open("traffic.json", "r") as traffic_file:
        traffic = json.load(traffic_file)
    
    keys = [k for k in traffic.keys()]
    time_prev = traffic[keys[0]]["ts"]

    # Set traffic light to red
    traci.trafficlight.setPhase(traci.trafficlight.getIDList()[0], 0)

    t = threading.Thread(target=run_simulation_bg)
    t.daemon = True
    t.start()

    expected_cars = None

    for i, k in enumerate(traffic):

        if expected_cars is None:
            prev_cars = expected_cars = [f"masina{v}" for v in traffic[k]["ids"]] 
        else:
            prev_cars = expected_cars
            expected_cars = [f"masina{v}" for v in traffic[k]["ids"]]

        print(f"Expected cars {expected_cars}")
        print(f"Current cars {traci.vehicle.getIDList()}")
        # clean_up(prev_cars, expected_cars)
          
        for v in traffic[k]["ids"]:
            current_car_to_add = f"masina{v}"

            if current_car_to_add not in prev_cars or i == 0:
                try:
                    traci.vehicle.add(current_car_to_add, "main", departPos="100")
                    # traci.vehicle.setStop(current_car_to_add, "s1", laneIndex=random.randint(0, 2), pos=170, duration=10000)
                except Exception as e:
                    pass

        if cmqQ != None:
            try:
                msg = cmqQ.get_nowait()
                print(f"Got message: {msg}")
                if msg == "r":
                    traci.trafficlight.setPhase(traci.trafficlight.getIDList()[0], 2)
                elif msg == "g":
                    traci.trafficlight.setPhase(traci.trafficlight.getIDList()[0], 0)
            except Exception as e:
                pass

        if "ts" in traffic[k]:
            time_now = traffic[k]["ts"]
            time.sleep(time_now - time_prev + 1)
            time_prev = time_now

    traci.close()

if __name__ == "__main__":
    traci.start(["sumo-gui", "-c", "unirii.sumocfg"])
    run()

def run_controllable_sumo(cmdQ):
    traci.start(["sumo-gui", "-c", "unirii.sumocfg", "--time-to-teleport", "-1"])
    run(cmdQ)