MODEL='model name folder'#'MAC_ReconNet' 
BASE_PATH='<base path>'
# Each Acquisition context has three settings - DATASET_TYPE, MASK_TYPE and acceleration factor represented in short ACC_FACTOR
# DATASET_TYPE indicates the different types of anatomy images that we would like to combine in a single training
# for example, 1. if we intend to combine T1 and FLAIR as two different data types of a subject then we could set this as DATASET_TYPE='mrbrain_t1','mrbrain_flair'. 2. If we wish to combine cardiac and brain anatomies in a single training we could set this as DATASET_TYPE='cardiac','brain'
DATASET_TYPE='mrbrain_t1','mrbrain_flair'
# MASK type indicates the kind of mask pattern which we use in the training. For example, if we use two mask types -Cartesian and Gaussian we set this as follows
MASK_TYPE='cartesian','gaussian'
# the acceleration factor for reconstruction is the third setting. The acceleration factor value along with 'x' character. This is set as follows.
ACC_FACTORS='4x','5x','8x'
BATCH_SIZE=4
NUM_EPOCHS=150
DEVICE='cuda:0'
EXP_DIR=${BASE_PATH}'/experiments/<path to where the model is saved>/'${MODEL}


#TRAIN_PATH=${BASE_PATH}'/datasets/'${DATASET_TYPE}'/'${MASK_TYPE}'/train/'
TRAIN_PATH=${BASE_PATH}'/datasets/'
VALIDATION_PATH=${BASE_PATH}'/datasets/'
USMASK_PATH=${BASE_PATH}'/<path where undersampling masks are stored>/us_masks/'
echo python train.py --batch-size ${BATCH_SIZE} --num-epochs ${NUM_EPOCHS} --device ${DEVICE} --exp-dir ${EXP_DIR} --train-path ${TRAIN_PATH} --validation-path ${VALIDATION_PATH} --dataset_type ${DATASET_TYPE} --usmask_path ${USMASK_PATH} --acceleration_factor ${ACC_FACTORS} --mask_type ${MASK_TYPE}
python train.py --batch-size ${BATCH_SIZE} --num-epochs ${NUM_EPOCHS} --device ${DEVICE} --exp-dir ${EXP_DIR} --train-path ${TRAIN_PATH} --validation-path ${VALIDATION_PATH} --dataset_type ${DATASET_TYPE} --usmask_path ${USMASK_PATH} --acceleration_factor ${ACC_FACTORS} --mask_type ${MASK_TYPE}
