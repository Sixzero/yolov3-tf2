source ../diabtrend-research/env/bin/activate 
export PYTHONPATH=$(pwd)
python tools/voc2012.py --output_path train/labelimg/tfrecords --data_dir train/labelimg

