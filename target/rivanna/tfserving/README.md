# Steps:

You must be on biihead, otherwise, the singularity commands are not whitelisted and will fail

```bash
cd target/rivanna/tfserving/train

# make clean

```

```bash
echo $OSMI_PROJECT_BASE # this must give you something.
# if it doesnt give you something, do the following.
export OSMI_PROJECT_BASE=/scratch/$USER/osmi2
# or whatever the name of your repo is, e.g. /scratch/$USER/osmi-bench-new
cd $OSMI_PROJECT_BASE
```

```bash
cd $OSMI_PROJECT_BASE/target/rivanna/tfserving/train
source env.sh
pip install -r requirements.txt
```

Dependencies should be installed with no error.

```bash
cd $OSMI_PROJECT_BASE/osmi-bench-new/target/rivanna/tfserving/train
time sh images.sh
cd images
time make images
cd ..
```
