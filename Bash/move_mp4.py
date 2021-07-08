import os

_path_root= '.'

paths = []
paths_new = []
for root,dirs,names in os.walk(_path_root):
    for name in names:
        ext=os.path.splitext(name)[1]
        if ext=='.mp4':
            fromdir=os.path.join(root,name)
            fromdir_new= os.path.join(
                fromdir.split("\\")[0],
                "_".join(fromdir.split("\\")[1:])
            )
            paths.append(fromdir)
            paths_new.append(fromdir_new)
            
            
for i in range(len(paths)):
    os.rename(paths[i],paths_new[i])
