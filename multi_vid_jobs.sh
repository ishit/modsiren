#!/bin/bash
echo "Starting worker $1 on $(hostname)"
echo "Testing GPU configuration..."
if ! nvidia-smi; then
    echo "nvidia-smi failed, aborting"
    echo $(hostname)
    exit
fi

model_id=$1
echo "model" $model_id 

# command line arguments:
#$1 -> model
#$2 -> video, 1 indexed
#$3 -> gpu

# Set root dirs
# DATA=/newfoundland2/ishit/
# ROOT=/home/ishit/temp_root/ishit
# OUTPUT=/newfoundland2/ishit/

DATA=/mnt/ilcompf9d1/user/mgharbi/data/ishit
ROOT=/mnt/ilcompf9d1/user/mgharbi/code/ishit
OUTPUT=/mnt/ilcompf9d1/user/mgharbi/output/ishit

# # # get code
# if [ ! -d $ROOT ]
# then
#  git clone https://github.com/ishit/localImplicit_release.git $ROOT
# else
#  cd $ROOT
#  git fetch --all
#  git reset --hard origin/master
#  cd ..
# fi

# move code and data to shared memory
rm -rf /dev/shm/ishit_vid
rm -rf /dev/shm/vid
rm -rf /dev/shm/ishit
rm -rf /dev/shm/demosaic_genetic_search
rm -rf /dev/shm/data
rm -rf /dev/shm/[0-9]
rm -rf /dev/shm/[0-9][0-9]

ls /dev/shm
du -h -d 1 /dev/shm

CODE=$ROOT
CODE_LOCAL=/dev/shm/ishit_vid/$1/ishit
mkdir -p $CODE_LOCAL

echo "Copying code and data to local memory drive"
echo $CODE $(dirname $CODE_LOCAL)
mkdir -p /dev/shm/$1
#rsync -av --ignore-existing $CODE $(dirname $CODE_LOCAL)
rsync -av $CODE $(dirname $CODE_LOCAL)

# install python modules
cd $CODE_LOCAL
echo "current dir" $(pwd)
echo "Installing python modules"
pip install -r requirements.txt
pip install lpips # didn't worky locally with requirements.txt!

# get config scripts
CONFIG_DIR=$CODE_LOCAL/config/multi_vid/

echo $CONFIG_DIR
configs=()
for c in "$CONFIG_DIR*.yml"
do
    configs+=($c)
done

let jobid=$model_id
curr_config="${configs[$jobid]}"
echo "Running job" $jobid on worker $1 with config $curr_config

cd $CODE_LOCAL

VID_PATH=bikes2/

# get data
echo "downloading data"
mkdir -p bikes_processed_2
rsync -av --ignore-existing $DATA/bikes2/ $VID_PATH
# rsync -av --ignore-existing $DATA/bikes_processed_2.zip .

# echo extracting data
# if [ ! -d "bikes_processed_2" ] 
# then
#     unzip bikes_processed_2.zip
# fi

echo config: $curr_config
echo Training
mkdir -p $OUTPUT

# edit config file
cd $CODE_LOCAL

LOG_DIR=$OUTPUT/logs_multi_vid
cp $curr_config temp_config.yml
echo $val_filenames
sed -i "s~\(log_dir *: *\).*~\1$LOG_DIR~" temp_config.yml
sed -i "s~\(vid_path *: *\).*~\1$VID_PATH~" temp_config.yml
sed -i "s~\(image_step *: *\).*~\1 1000~" temp_config.yml
sed -i "s~\(scalars_step *: *\).*~\1 1000~" temp_config.yml

python -m scripts.train_multi_vid --config temp_config.yml
echo "job done"
