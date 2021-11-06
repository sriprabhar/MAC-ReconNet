MODEL='model name folder'#'MAC_ReconNet' 
BASE_PATH='<base path>'

echo ${MODEL}

for DATASET_TYPE in 'mrbrain_t1' 'mrbrain_flair'
    do
    for MASK_TYPE in 'cartesian' 'gaussian'
        do 
        for ACC_FACTOR in '4x' '5x' '8x'
            do 
            echo ${DATASET_TYPE}','${MASK_TYPE}','${ACC_FACTOR} 

            REPORT_PATH=${BASE_PATH}'/experiments/<path to where model folder is present>/'${MODEL}'/results/'${DATASET_TYPE}'/'${MASK_TYPE}'/acc_'${ACC_FACTOR}'/report_'${DATASET_TYPE}'_'${MASK_TYPE}'_'${ACC_FACTOR}'.txt'
            echo ${REPORT_PATH}
            cat ${REPORT_PATH}
            echo "\n"
            done 
        done 
    done 

