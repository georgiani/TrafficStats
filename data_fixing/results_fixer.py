import json

def is_last_element(el, lst):
    return lst[-1] == el

def is_first_element(el, lst):
    return lst[0] == el

if __name__ == "__main__":
    files = [
        "res_tuesday_1", 
        "res_tuesday_2", 
        "res_wed_0", 
        "res_wed_1", 
        "res_wed_2", 
        "res_wed_3", 
        "res_wed_4", 
        "res_thu_0", 
        "res_thu_1", 
        "res_thu_2", 
        "res_thu_3_night", 
        "res_friday_0", 
        "res_friday_1", 
        "res_saturday_0", 
        "res_sunday_1", 
        "res_sunday_2"
    ]
    tolerance = 2 # 10 seconds
    for filename in files:
        with open(f"date/{filename}.json", "r") as f:
            results = json.load(f)
            
            cars = {}
            
            # Keys are not necessarily alligned with the indexes
            keys = list(results.keys())
            for i, k in enumerate(keys):
                # Take ids in order
                ids = results[k]["ids"]

                # Go throuh the ids
                for car_id in ids:
                    # if it already was previously in another timestamp ids then process it
                    if car_id in cars:
                        # If the distance from the last appearance is under the tolerance then we can
                        #   fill between since it's very probable that it's the same car
                        distance = i - cars[car_id]

                        # Distance > 1 in order to ignore consecutive appearances
                        if distance <= tolerance and distance > 1: 
                            # Fill from where it last appeared to where it reappears
                            for j in range(cars[car_id] + 1, i):
                                results[keys[j]]["ids"].append(car_id)
                            
                    # This is now the last place where it appeared since it could
                    #   disappear again
                    cars[car_id] = i
        with open(f"date_fixed/{filename}_fixed.json", "w") as new_f:
            new_f.write(json.dumps(results))
