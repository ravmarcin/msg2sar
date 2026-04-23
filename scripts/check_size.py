import os


def get_dir_size(directory):
    # assign size
    size = 0

    # get size
    for path, dirs, files in os.walk(directory):
        for f in files:
            fp = os.path.join(path, f)
            size += os.path.getsize(fp)
    return size


    
directory = '/home/rav_marcin/projects/msg2sar/data/sar/sbas/desc/2023/bogo_pl/data'

for f in os.listdir(directory):
    f_path = os.path.join(directory, f)
    # display size
    print(f"{f}: {(get_dir_size(f_path)  >> 20)} MB") 