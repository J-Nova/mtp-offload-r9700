tmux new -d -s retrain 'bash /home/juup/radiance-vllm-mxfp4/paroquant/drafter/retrain_r2.sh 2>&1 | tee -a ~/drafter_ft/r2/run.log'
