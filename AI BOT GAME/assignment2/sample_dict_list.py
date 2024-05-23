sample = dict()
data = list()
global_index = 0
global_key = 0
global_pending = 0

for i in range(1, 4):
    data.append(i*10)

data_len = len(data)
for itr, key in enumerate(data):
    sample[key] = list()
    vals = sample[key]
    for val in data:
        if key != val:
            vals.append(val)

print(sample, data)

def remove_vals_2(x):
    if x not in data:
        return

    global global_index, global_key, global_pending
    index = data.index(x)
    if global_pending == 1:
        del sample[x]
    else:
        if (index > global_index):
            index -= 1
        else:
            global_index -= 1
        sample[global_key].pop(index)
        if len(sample[global_key]) == 0:
            del sample[global_key]

    data.remove(x)

def remove_vals(x):
    if x not in data:
        return

    index = data.index(x)
    print(index)
    for itr, key in enumerate(data):
        if itr < index:
            sample[key].pop(index - 1)
        elif itr > index:
            sample[key].pop(index)
        else:
            del sample[key]

    data.remove(x)

def retain_success_vals(x, val = 1):
    if x not in data:
        return

    index = data.index(x)
    if val == 1:
        for itr, key in enumerate(data):
            final_index = index
            if itr < index:
                final_index = index - 1
            elif itr ==  index:
                del sample[key]
                continue
            my_list = sample[key]
            for val_itr in range(len(data) - 1):
                if val_itr < final_index:
                    my_list.pop(0)
                elif val_itr > final_index:
                    my_list.pop(1)
    else:
        for itr, key in enumerate(data):
            if index != itr:
                del sample[key]

    data.pop(index)
    global global_index, global_key, global_pending
    global_index = index
    global_key = x
    global_pending = val

remove_vals(10)
print(sample, data)
# remove_vals(20)
# print(sample, data)
# remove_vals(80)
# print(sample, data)
# remove_vals(40)
# print(sample, data)
retain_success_vals(30, 1)
print(sample, data)
# remove_vals_2(70)
# print(sample, data)
# remove_vals_2(50)
# print(sample, data)
# remove_vals_2(30)
# print(sample, data)
# remove_vals_2(70)
# print(sample, data)
