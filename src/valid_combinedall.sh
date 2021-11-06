MODEL='model name folder'#'MAC_ReconNet' 
BASE_PATH='<base path>'

CHECKPOINT=${BASE_PATH}'/experiments/<path to where the model folder is saved>/'${MODEL}'/best_model.pt'
BATCH_SIZE=1
DEVICE='cuda:0'

for DATASET_TYPE in 'mrbrain_t1' 'mrbrain_flair'
    do
    for MASK_TYPE in 'cartesian' 'gaussian'
        do 
        for ACC_FACTOR in '4x' '5x' '8x'
            do 
            echo ${DATASET_TYPE}','${MASK_TYPE}','${ACC_FACTOR} 
            OUT_DIR=${BASE_PATH}'/experiments/<path to where model folder is present>/'${MODEL}'/results/'${DATASET_TYPE}'/'${MASK_TYPE}'/acc_'${ACC_FACTOR}
            DATA_PATH=${BASE_PATH}'/datasets/'${DATASET_TYPE}'/'${MASK_TYPE}'/validation/acc_'${ACC_FACTOR}
            echo python valid.py --checkpoint ${CHECKPOINT} --out-dir ${OUT_DIR} --batch-size ${BATCH_SIZE} --device ${DEVICE} --data-path ${DATA_PATH} --acceleration_factor ${ACC_FACTOR} --dataset_type ${DATASET_TYPE} --mask_type ${MASK_TYPE} 
            python valid.py --checkpoint ${CHECKPOINT} --out-dir ${OUT_DIR} --batch-size ${BATCH_SIZE} --device ${DEVICE} --data-path ${DATA_PATH} --acceleration_factor ${ACC_FACTOR} --dataset_type ${DATASET_TYPE} --mask_type ${MASK_TYPE} 
            done 
        done 
    done



