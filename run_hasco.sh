cd ./src
python main.py --optim_obj EDP --population 20 --epochs 20  --accelerator $2 --mapper Saturn --type arch --version v1 --workload $1 --layer_id 0 --batch_size 1 --all_DNN --search_mode HASCO
cd ..