import os
import shutil
current_path=os.getcwd()
print(current_path)
new_path=os.getcwd()+"\\can_train"
file_names = next(os.walk(current_path))[2]
possible_names = []
for i in range(len(file_names)):
    if 'S' not in file_names[i] and 'filter_side_photos.py' not in file_names[i]:
        shutil.move(current_path+'\\'+file_names[i],new_path+'\\'+file_names[i])


