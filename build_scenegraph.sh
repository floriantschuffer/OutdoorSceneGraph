#!/bin/bash
PROJECT_PATH="/Users/floriantschuffer/huggingFaceVenv/src/OutdoorSceneGraph"
SCENE="bag_2025_6_4_15_27_55"
SCENE="CAB"
CONDA_BIN="/Users/floriantschuffer/huggingFaceVenv/.venv/bin"

# source $CONDA_BIN/activate

cd $PROJECT_PATH
cd src

python build_objects.py --path $PROJECT_PATH --scene $SCENE #--show_runtimes
python build_descriptions.py --path $PROJECT_PATH --scene $SCENE
python build_edges.py --path $PROJECT_PATH --scene $SCENE