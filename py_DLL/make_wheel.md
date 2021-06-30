

A simple `setup.py`

```python
import os
from setuptools import setup, find_packages

doclines = __doc__.split('\n')

with open(os.path.join(os.path.dirname(__file__), 'requirements.txt')) as f:
    requirements = f.readlines()

setup(
    name='xxx',
    author='xxx',
    author_email='xxx@gmail.com',
    license='http://www.gnu.org/licenses/gpl-2.0.html',
    platforms=['ANY'],

    provides=['xxx'],
    packages=find_packages(),
    description=doclines[0],
    long_description='\n'.join(doclines[2:]),
    classifiers=classifications.split('\n'),
)
```

then run

```bash
python setup.py bdist_wheel
```

and you get `xxx.whl` in `.\dist`.



The whl file can be used to installation via

```bash
pip install xxx.whl
```

