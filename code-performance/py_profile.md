run cProfile in python

```python
import cProfile
cProfile.run('your_func()','profile.prof')
```

in CMD, run

```bash
snakeviz profile.prof
```
which start a web server to visualize profile.
