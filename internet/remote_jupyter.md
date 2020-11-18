

get hash key of your password

```
from notebook.auth import passwd
passwd()
```

for example, `'sha1:128c0c7eac38:49aa851556830d3d2cbcf70d31e139822c9fd36e'` 



My preference

```python
c.NotebookApp.notebook_dir = r'D:\User'
c.NotebookApp.allow_remote_access = True
c.NotebookApp.open_browser = False
c.NotebookApp.ip = '*'
c.NotebookApp.password = 'sha1:128c0c7eac38:49aa851556830d3d2cbcf70d31e139822c9fd36e'
c.NotebookApp.port = 2333
```

