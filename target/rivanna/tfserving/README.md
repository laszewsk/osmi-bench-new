

```bash
cd target/rivanna/tfserving/train

# make clean

```

```bash
echo $OSMI_PROJECT_BASE # this must give you something.
# if it doesnt give you something, do the following.
export OSMI_PROJECT_BASE=/scratch/$USER/osmi2
mkdir -p $OSMI_PROJECT_BASE
cd $OSMI_PROJECT_BASE
```

must be non empty


cd $OSMI_PROJECT_BASE/target/rivanna/train
source env.sh

pip install -r requirements.txt
