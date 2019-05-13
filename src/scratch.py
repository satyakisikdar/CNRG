import time
from multiprocessing.pool import Pool

min_val = float('inf')
min_item = None

def update_min(item):
    global min_val, min_item

    print('outside if', min_val, min_item)
    if item[0] < min_val:
        print(f'updatin min from {min_val} to {item[0]}')
        min_val = item[0]
        min_item = item[1]

    time.sleep(0.5)




if __name__ == '__main__':
    lst = [(4, 'a'), (2, 'b'), (1, 'c'), (0, 'd'), (3, 'f')]
    pool = Pool(processes=4)
    pool.map(update_min, lst)
    pool.close()
    pool.join()
    print(min_item, min_val)