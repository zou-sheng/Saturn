cd ./src
python main.py --optim_obj EDP --population 20 --epochs 20  --accelerator NEW --mapper Saturn --type arch --version v1 --workload llama2_7B --layer_id 0 --batch_size 1 --all_DNN --search_mode HF
cd ..