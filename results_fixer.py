import json

def is_last_element(el, lst):
    return lst[-1] == el

def is_first_element(el, lst):
    return lst[0] == el

if __name__ == "__main__":
    with open("res.json", "r") as f:
        results = json.load(f)

        keys = list(results.keys())
        for i, k in enumerate(keys):
            ids = results[k]["ids"]
            
            if i > 0 and i != len(keys) - 1:
                prev_ids = results[keys[i - 1]]["ids"]
                next_ids = results[keys[i + 1]]["ids"]
                
                for pid in prev_ids:
                    if pid not in ids and pid in next_ids:
                        results[k]["ids"].append(pid)

    with open("res_modified.json", "w") as new_f:
        new_f.write(json.dumps(results))
