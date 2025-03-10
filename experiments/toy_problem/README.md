original code from https://github.com/Cranial-XIX/FAMO/tree/main/experiments/toy

Famo, 2023

set pythonpath

export PYTHONPATH=$(pwd)

run command

```
python toy_problem/main.py  -m interest_function=f1,f2,f12 optimization=alignedmtl,cagrad,famo,graddrop,l2bmgrad,logbmgrad,ls,minbmgrad,pcgrad
```

run command for thopology study

```
python toy_problem/main.py --config-name=config_thopology

```
