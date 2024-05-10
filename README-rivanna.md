
## Steps that do not have to repeated

1. The reposotory has been imported from the original location at

This allows that the history is preserved

2. A README-rivanna.md which is this file has been added so that we can record how another user can use it on rivanna

## Steps that may be unclear in future

3. We have not explored what happens when the original repository gets modified and  how to stay in sync.
   The original import was conducted on May 10, 2024

Links


Discord: Not sure what the link actually is, Gregor needs help on this


This is a working discord with limited membership. ITs for writing a
paper for AI workflows based on cloudmesh-ee. Authorship in paper is
not automatic if you are member of this group. You will have to
significantly contribut in time and intellectual contributions.

* Repo is at https://github.com/laszewsk/osmi-bench-new
* Original is at https://code.ornl.gov/whb/osmi-bench
* Gregors code that we can leverage is at https://github.com/laszewsk/osmi-bench

* Aoom link, contact Gregor


## One time setup Setup

```bash
export PROJECT_BASE=/scratch/$USER/osmi2
mkdir -p $PROJECT_BASE
cd $PROJECT_BASE


git clone https://github.com/laszewsk/osmi-bench-new

mkdir wes
cd wes
git clone https://code.ornl.gov/whb/osmi-bench
cd ..

mkdir gregor
cd gregor
git clone https://github.com/laszewsk/osmi-bench
cd ..
```

## Editing

We prefer that code is edited with VSCODE using the remote
features. We dissallow for this project the usage of UVA Ondemand and
FASTX as well as jupyter started via ondemand. All code must be edited
with vscode. it is allowed to use emacs and pycharm. code formating
rules need to be established. Gregor needs help with that so we can do
uniform code formatting.