MODEL='model name folder'#'MAC_ReconNet' 
BASE_PATH='<base path>'

#DATASET_TYPE='mrbrain_t1'
DATASET_TYPE='mrbrain_flair'
MASK_TYPE='cartesian'

#<<ACC_FACTOR_5x
ACC_FACTOR='5x'
TARGET_PATH=${BASE_PATH}'/datasets/'${DATASET_TYPE}'/'${MASK_TYPE}'/validation/acc_'${ACC_FACTOR}

PREDICTIONS_PATH=${BASE_PATH}'/experiments/<path to where model folder is present>/'${MODEL}'/results/'${DATASET_TYPE}'/'${MASK_TYPE}'/acc_'${ACC_FACTOR}'/'

REPORT_PATH=${BASE_PATH}'/experiments/<path to where model folder is present>/'${MODEL}'/results/'${DATASET_TYPE}'/'${MASK_TYPE}'/acc_'${ACC_FACTOR}'/'
python measures_csv.py --target-path ${TARGET_PATH} --predictions-path ${PREDICTIONS_PATH} --report-path ${REPORT_PATH} --acc-factor ${ACC_FACTOR}
#ACC_FACTOR_5x


