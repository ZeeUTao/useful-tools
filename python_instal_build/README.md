The folder includes python DLL files and related installing files. 



When you import some package, including sqlite, openSSL ... in python 3.7.x, 
some errors ras: ImportError: DLL load failed:

```
ImportError: DLL load failed:
```

So we can put the .dll file in \anaconda\DLLs\ to fix it



Jupyter Notebook命令行启动报错: DLL load failed
解决方法：
Path添加环境变量
anaconda\
anaconda\Script\
anaconda\Library\bin\
