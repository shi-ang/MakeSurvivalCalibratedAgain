#!/bin/bash

#export PYTHONPATH="${PYTHONPATH}:/home/shiang/Documents/Survival_Prediction/ConformalSurvDist/SurvivalEVAL"

mono_type=bootstrap # change it to ceil/bootstrap, ceil for SEER, and bootstrap for others
batch_size=256
lr=0.001

for data in HFCR PBC WHAS500;
  do
    for model in MTLR DeepHit CoxPH AFT GB CoxTime CQRNN;
      do
        python3 run_baseline.py --data $data --model $model --n_quantiles 9 --mono_method $mono_type --batch_size $batch_size --lr $lr

        for pp in CSD CSD-iPOT;
          do
            for dm in sampling;
              do
#                python3 run.py --data $data --model $model --post_process $pp --use_train False --n_quantiles 9 --decensor_method $dm --mono_method $mono_type --batch_size $batch_size --lr $lr
                python3 run.py --data $data --model $model --post_process $pp --use_train True --n_quantiles 9 --decensor_method $dm --mono_method $mono_type --batch_size $batch_size --lr $lr
#                python3 run.py --data $data --model $model --post_process $pp --use_train False --n_quantiles 19 --decensor_method $dm --mono_method $mono_type --batch_size $batch_size --lr $lr
#                python3 run.py --data $data --model $model --post_process $pp --use_train True --n_quantiles 19 --decensor_method $dm --mono_method $mono_type --batch_size $batch_size --lr $lr
#                python3 run.py --data $data --model $model --post_process $pp --use_train True --n_quantiles 39 --decensor_method $dm --mono_method $mono_type --batch_size $batch_size --lr $lr
#                python3 run.py --data $data --model $model --post_process $pp --use_train True --n_quantiles 49 --decensor_method $dm --mono_method $mono_type --batch_size $batch_size --lr $lr
#                python3 run.py --data $data --model $model --post_process $pp --use_train True --n_quantiles 99 --decensor_method $dm --mono_method $mono_type --batch_size $batch_size --lr $lr
#                python3 run.py --data $data --model $model --post_process $pp --use_train True --n_quantiles 9 --decensor_method $dm --mono_method $mono_type --batch_size $batch_size --lr $lr --n_sample 100
#                python3 run.py --data $data --model $model --post_process $pp --use_train True --n_quantiles 9 --decensor_method $dm --mono_method $mono_type --batch_size $batch_size --lr $lr --n_sample 10
#                python3 run.py --data $data --model $model --post_process $pp --use_train True --n_quantiles 9 --decensor_method $dm --mono_method $mono_type --batch_size $batch_size --lr $lr --n_sample 5
#                python3 run.py --data $data --model $model --post_process $pp --use_train True --n_quantiles 9 --decensor_method $dm --mono_method $mono_type --batch_size $batch_size --lr $lr --n_sample 3
              done
          done
      done
  done

