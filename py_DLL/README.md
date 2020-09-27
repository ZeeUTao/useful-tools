The folder includes python DLL files and related installing files. 



When you import some package, including sqlite, openSSL ... in python 3.7.x, 
some errors ras: ImportError: DLL load failed:

```
ImportError: DLL load failed:
```

So we can put the .dll file in \anaconda\DLLs\ to fix it
