source ~/.bashrc
source /storage/agrp/rotemdo/venv/bin/activate
IOTHROTTLE_LIMIT=0
nvidia-smi
cd $RUN_DIR
CMD="python -m physics.g4h_ionisation.generators.train_networks.continuous_generator.train"

echo $CMD
eval $CMD
