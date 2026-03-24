cd ./src
# python main.py --optim_obj EDP --population 100 --epochs 100  --accelerator Simba --mapper Saturn --type arch --version v1 --workload alexnet --layer_id 0 --batch_size 1 --expanded_scope 0.2
# python main.py --optim_obj EDP --population 100 --epochs 100  --accelerator Simba --mapper Saturn --type arch --version v1 --workload alexnet --layer_id 0 --batch_size 1 --expanded_scope 0.2 --all_DNN
# python main.py --optim_obj EDP --population 20 --epochs 20  --accelerator Gemmini --mapper Saturn --type arch --version v1 --workload bert --layer_id 0 --batch_size 1 --expanded_scope 0.2 --all_DNN
# python main.py --optim_obj EDP --population 20 --epochs 50  --accelerator Gemmini --mapper Saturn --type arch --version v1 --workload retinanet --layer_id 0 --batch_size 1 --expanded_scope 0.2 --all_DNN
# python main.py --optim_obj EDP --population 20 --epochs 100  --accelerator Gemmini --mapper Saturn --type arch --version v1 --workload resnet50 --layer_id 0 --batch_size 1 --expanded_scope 0.44 --all_DNN
# python main.py --optim_obj EDP --population 20 --epochs 50  --accelerator Gemmini --mapper Saturn --type arch --version v1 --workload alexnet --layer_id 0 --batch_size 1 --expanded_scope 0.2 --all_DNN
# python main.py --optim_obj EDP --population 20 --epochs 200  --accelerator Gemmini --mapper Saturn --type arch --version v1 --workload vgg16 --layer_id 0 --batch_size 1 --expanded_scope 0.2 --all_DNN
# python main.py --optim_obj EDP --population 20 --epochs 50  --accelerator Gemmini --mapper Saturn --type arch --version v1 --workload bert --layer_id 0 --batch_size 1 --expanded_scope 0.2 --all_DNN
# python main.py --optim_obj EDP --population 20 --epochs 50  --accelerator Gemmini --mapper Saturn --type arch --version v1 --workload unet --layer_id 0 --batch_size 1 --expanded_scope 0.2 --all_DNN
# python main.py --optim_obj EDP --population 20 --epochs 50  --accelerator Gemmini --mapper Saturn --type arch --version v1 --workload vgg16 --layer_id 0 --batch_size 1 --all_DNN
python main.py --optim_obj EDP --population 30 --epochs 50  --accelerator $2 --mapper Saturn --type arch --version v1 --workload $1 --layer_id 0 --batch_size 1 --expanded_scope 0.2 --all_DNN --search_mode MF --cluster_num 10
# python main.py --optim_obj EDP --population 20 --epochs 50  --accelerator Gemmini --mapper Saturn --type arch --version v1 --workload llama2_13B --layer_id 0 --batch_size 1 --expanded_scope 0.2 --all_DNN
# python main.py --optim_obj EDP --population 20 --epochs 100  --accelerator Gemmini --mapper Saturn --type arch --version v1 --workload llama2_34B --layer_id 0 --batch_size 1 --expanded_scope 0.2 --all_DNN
# echo "Random-llama2_34B."
# python main.py --optim_obj EDP --population 20 --epochs 50  --accelerator Gemmini --mapper Random --type arch --version v1 --workload llama2_34B --layer_id 0 --batch_size 1 --expanded_scope 0.2 --all_DNN
# echo "Random-llama2_13B."
# python main.py --optim_obj EDP --population 20 --epochs 50  --accelerator Gemmini --mapper Random --type arch --version v1 --workload llama2_13B --layer_id 0 --batch_size 1 --expanded_scope 0.2 --all_DNN
# echo "Random-llama2_7B."
# python main.py --optim_obj EDP --population 20 --epochs 50  --accelerator Gemmini --mapper Random --type arch --version v1 --workload llama2_7B --layer_id 0 --batch_size 1 --expanded_scope 0.2 --all_DNN
# echo "Random-retinanet."
# python main.py --optim_obj EDP --population 20 --epochs 50  --accelerator Gemmini --mapper Random --type arch --version v1 --workload retinanet --layer_id 0 --batch_size 1 --expanded_scope 0.2 --all_DNN
# echo "Random-unet."
# python main.py --optim_obj EDP --population 20 --epochs 50  --accelerator Gemmini --mapper Random --type arch --version v1 --workload unet --layer_id 0 --batch_size 1 --expanded_scope 0.2 --all_DNN
# echo "Random-bert."
# python main.py --optim_obj EDP --population 20 --epochs 50  --accelerator Gemmini --mapper Random --type arch --version v1 --workload bert --layer_id 0 --batch_size 1 --expanded_scope 0.2 --all_DNN
# echo "Random-vgg16."
# python main.py --optim_obj EDP --population 20 --epochs 50  --accelerator Gemmini --mapper Random --type arch --version v1 --workload vgg16 --layer_id 0 --batch_size 1 --expanded_scope 0.2 --all_DNN
# echo "Random-alexnet."
# python main.py --optim_obj EDP --population 20 --epochs 50  --accelerator Gemmini --mapper Random --type arch --version v1 --workload alexnet --layer_id 0 --batch_size 1 --expanded_scope 0.2 --all_DNN
# echo "Random-resnet50."
# python main.py --optim_obj EDP --population 20 --epochs 50  --accelerator Gemmini --mapper Random --type arch --version v1 --workload resnet50 --layer_id 0 --batch_size 1 --expanded_scope 0.2 --all_DNN
# echo "BO-llama2_34B."
# python main.py --optim_obj EDP --population 20 --epochs 50  --accelerator Gemmini --mapper BO --type arch --version v1 --workload llama2_34B --layer_id 0 --batch_size 1 --expanded_scope 0.2 --all_DNN
# echo "BO-llama2_13B."
# python main.py --optim_obj EDP --population 20 --epochs 50  --accelerator Gemmini --mapper BO --type arch --version v1 --workload llama2_13B --layer_id 0 --batch_size 1 --expanded_scope 0.2 --all_DNN
# echo "BO-llama2_7B."
# python main.py --optim_obj EDP --population 20 --epochs 50  --accelerator Gemmini --mapper BO --type arch --version v1 --workload llama2_7B --layer_id 0 --batch_size 1 --expanded_scope 0.2 --all_DNN
# echo "BO-retinanet."
# python main.py --optim_obj EDP --population 20 --epochs 50  --accelerator Gemmini --mapper BO --type arch --version v1 --workload retinanet --layer_id 0 --batch_size 1 --expanded_scope 0.2 --all_DNN
# echo "BO-unet."
# python main.py --optim_obj EDP --population 20 --epochs 50  --accelerator Gemmini --mapper BO --type arch --version v1 --workload unet --layer_id 0 --batch_size 1 --expanded_scope 0.2 --all_DNN
# echo "BO-bert."
# python main.py --optim_obj EDP --population 20 --epochs 50  --accelerator Gemmini --mapper BO --type arch --version v1 --workload bert --layer_id 0 --batch_size 1 --expanded_scope 0.2 --all_DNN
# echo "BO-vgg16."
# python main.py --optim_obj EDP --population 20 --epochs 50  --accelerator Gemmini --mapper BO --type arch --version v1 --workload vgg16 --layer_id 0 --batch_size 1 --expanded_scope 0.2 --all_DNN
# echo "BO-alexnet."
# python main.py --optim_obj EDP --population 20 --epochs 50  --accelerator Gemmini --mapper BO --type arch --version v1 --workload alexnet --layer_id 0 --batch_size 1 --expanded_scope 0.2 --all_DNN
# echo "BO-resnet50."
# python main.py --optim_obj EDP --population 20 --epochs 50  --accelerator Gemmini --mapper BO --type arch --version v1 --workload resnet50 --layer_id 0 --batch_size 1 --expanded_scope 0.2 --all_DNN
# echo "Random-llama2_34B."
# python main.py --optim_obj EDP --population 20 --epochs 50  --accelerator Simba --mapper Random --type arch --version v1 --workload llama2_34B --layer_id 0 --batch_size 1 --expanded_scope 0.2 --all_DNN
# echo "Random-llama2_13B."
# python main.py --optim_obj EDP --population 20 --epochs 50  --accelerator Simba --mapper Random --type arch --version v1 --workload llama2_13B --layer_id 0 --batch_size 1 --expanded_scope 0.2 --all_DNN
# echo "Random-llama2_7B."
# python main.py --optim_obj EDP --population 20 --epochs 50  --accelerator Simba --mapper Random --type arch --version v1 --workload llama2_7B --layer_id 0 --batch_size 1 --expanded_scope 0.2 --all_DNN
# echo "Random-retinanet."
# python main.py --optim_obj EDP --population 20 --epochs 50  --accelerator Simba --mapper Random --type arch --version v1 --workload retinanet --layer_id 0 --batch_size 1 --expanded_scope 0.2 --all_DNN
# echo "Random-unet."
# python main.py --optim_obj EDP --population 20 --epochs 50  --accelerator Simba --mapper Random --type arch --version v1 --workload unet --layer_id 0 --batch_size 1 --expanded_scope 0.2 --all_DNN
# echo "Random-bert."
# python main.py --optim_obj EDP --population 20 --epochs 50  --accelerator Simba --mapper Random --type arch --version v1 --workload bert --layer_id 0 --batch_size 1 --expanded_scope 0.2 --all_DNN
# echo "Random-vgg16."
# python main.py --optim_obj EDP --population 20 --epochs 50  --accelerator Simba --mapper Random --type arch --version v1 --workload vgg16 --layer_id 0 --batch_size 1 --expanded_scope 0.2 --all_DNN
# echo "Random-alexnet."
# python main.py --optim_obj EDP --population 20 --epochs 50  --accelerator Simba --mapper Random --type arch --version v1 --workload alexnet --layer_id 0 --batch_size 1 --expanded_scope 0.2 --all_DNN
# echo "Random-resnet50."
# python main.py --optim_obj EDP --population 20 --epochs 50  --accelerator Simba --mapper Random --type arch --version v1 --workload resnet50 --layer_id 0 --batch_size 1 --expanded_scope 0.2 --all_DNN
# echo "BO-llama2_34B."
# python main.py --optim_obj EDP --population 20 --epochs 50  --accelerator Simba --mapper BO --type arch --version v1 --workload llama2_34B --layer_id 0 --batch_size 1 --expanded_scope 0.2 --all_DNN
# echo "BO-llama2_13B."
# python main.py --optim_obj EDP --population 20 --epochs 50  --accelerator Simba --mapper BO --type arch --version v1 --workload llama2_13B --layer_id 0 --batch_size 1 --expanded_scope 0.2 --all_DNN
# echo "BO-llama2_7B."
# python main.py --optim_obj EDP --population 20 --epochs 50  --accelerator Simba --mapper BO --type arch --version v1 --workload llama2_7B --layer_id 0 --batch_size 1 --expanded_scope 0.2 --all_DNN
# echo "BO-retinanet."
# python main.py --optim_obj EDP --population 20 --epochs 50  --accelerator Simba --mapper BO --type arch --version v1 --workload retinanet --layer_id 0 --batch_size 1 --expanded_scope 0.2 --all_DNN
# echo "BO-unet."
# python main.py --optim_obj EDP --population 20 --epochs 50  --accelerator Simba --mapper BO --type arch --version v1 --workload unet --layer_id 0 --batch_size 1 --expanded_scope 0.2 --all_DNN
# echo "BO-bert."
# python main.py --optim_obj EDP --population 20 --epochs 50  --accelerator Simba --mapper BO --type arch --version v1 --workload bert --layer_id 0 --batch_size 1 --expanded_scope 0.2 --all_DNN
# echo "BO-vgg16."
# python main.py --optim_obj EDP --population 20 --epochs 50  --accelerator Simba --mapper BO --type arch --version v1 --workload vgg16 --layer_id 0 --batch_size 1 --expanded_scope 0.2 --all_DNN
# echo "BO-alexnet."
# python main.py --optim_obj EDP --population 20 --epochs 50  --accelerator Simba --mapper BO --type arch --version v1 --workload alexnet --layer_id 0 --batch_size 1 --expanded_scope 0.2 --all_DNN
# echo "BO-resnet50."
# python main.py --optim_obj EDP --population 20 --epochs 50  --accelerator Simba --mapper BO --type arch --version v1 --workload resnet50 --layer_id 0 --batch_size 1 --expanded_scope 0.2 --all_DNN
# python main.py --optim_obj EDP --population 20 --epochs 100  --accelerator Simba --mapper Saturn --type arch --version v1 --workload bert --layer_id 0 --batch_size 1 --expanded_scope 0.2 --all_DNN
# python main.py --optim_obj EDP --population 20 --epochs 100  --accelerator Simba --mapper Saturn --type arch --version v1 --workload retinanet --layer_id 0 --batch_size 1 --expanded_scope 0.2 --all_DNN
# python main.py --optim_obj EDP --population 20 --epochs 200  --accelerator Simba --mapper Saturn --type arch --version v1 --workload resnet50 --layer_id 0 --batch_size 1 --expanded_scope 0.44 --all_DNN
# python main.py --optim_obj EDP --population 20 --epochs 100  --accelerator Simba --mapper Saturn --type arch --version v1 --workload alexnet --layer_id 0 --batch_size 1 --expanded_scope 0.2 --all_DNN
# python main.py --optim_obj EDP --population 20 --epochs 100  --accelerator Simba --mapper Saturn --type arch --version v1 --workload vgg16 --layer_id 0 --batch_size 1 --expanded_scope 0.2 --all_DNN
# python main.py --optim_obj EDP --population 20 --epochs 100  --accelerator Simba --mapper Saturn --type arch --version v1 --workload bert --layer_id 0 --batch_size 1 --expanded_scope 0.2 --all_DNN
# python main.py --optim_obj EDP --population 20 --epochs 100  --accelerator Simba --mapper Saturn --type arch --version v1 --workload unet --layer_id 0 --batch_size 1 --expanded_scope 0.2 --all_DNN
# python main.py --optim_obj EDP --population 20 --epochs 100  --accelerator Simba --mapper Saturn --type arch --version v1 --workload retinanet --layer_id 0 --batch_size 1 --expanded_scope 0.2 --all_DNN
# python main.py --optim_obj EDP --population 20 --epochs 100  --accelerator Simba --mapper Saturn --type arch --version v1 --workload llama2_7B --layer_id 0 --batch_size 1 --expanded_scope 0.2 --all_DNN
# python main.py --optim_obj EDP --population 20 --epochs 100  --accelerator Simba --mapper Saturn --type arch --version v1 --workload llama2_13B --layer_id 0 --batch_size 1 --expanded_scope 0.2 --all_DNN
# python main.py --optim_obj EDP --population 20 --epochs 100  --accelerator Simba --mapper Saturn --type arch --version v1 --workload llama2_34B --layer_id 0 --batch_size 1 --expanded_scope 0.2 --all_DNN
# echo "-----------------------------------------------87"
# python main.py --optim_obj EDP --population 20 --epochs 100  --accelerator Simba --mapper Saturn --type arch --version v1 --workload densenet161 --layer_id 0 --batch_size 1 --expanded_scope 0.2 --all_DNN --weight_num 87
# echo "-----------------------------------------------80"
# python main.py --optim_obj EDP --population 20 --epochs 100  --accelerator Simba --mapper Saturn --type arch --version v1 --workload densenet161 --layer_id 0 --batch_size 1 --expanded_scope 0.2 --all_DNN --weight_num 80 
# echo "-----------------------------------------------70"
# python main.py --optim_obj EDP --population 20 --epochs 100  --accelerator Simba --mapper Saturn --type arch --version v1 --workload densenet161 --layer_id 0 --batch_size 1 --expanded_scope 0.2 --all_DNN --weight_num 70 
# echo "-----------------------------------------------60"
# python main.py --optim_obj EDP --population 20 --epochs 100  --accelerator Simba --mapper Saturn --type arch --version v1 --workload densenet161 --layer_id 0 --batch_size 1 --expanded_scope 0.2 --all_DNN --weight_num 60 
# echo "-----------------------------------------------50"
# python main.py --optim_obj EDP --population 20 --epochs 100  --accelerator Simba --mapper Saturn --type arch --version v1 --workload densenet161 --layer_id 0 --batch_size 1 --expanded_scope 0.2 --all_DNN --weight_num 50 
# echo "-----------------------------------------------40"
# python main.py --optim_obj EDP --population 20 --epochs 100  --accelerator Simba --mapper Saturn --type arch --version v1 --workload densenet161 --layer_id 0 --batch_size 1 --expanded_scope 0.2 --all_DNN --weight_num 40 
# echo "-----------------------------------------------30"
# python main.py --optim_obj EDP --population 20 --epochs 100  --accelerator Simba --mapper Saturn --type arch --version v1 --workload densenet161 --layer_id 0 --batch_size 1 --expanded_scope 0.2 --all_DNN --weight_num 30 
# echo "-----------------------------------------------20"
# python main.py --optim_obj EDP --population 20 --epochs 100  --accelerator Simba --mapper Saturn --type arch --version v1 --workload densenet161 --layer_id 0 --batch_size 1 --expanded_scope 0.2 --all_DNN --weight_num 20
# echo "-----------------------------------------------10"
# python main.py --optim_obj EDP --population 20 --epochs 100  --accelerator Simba --mapper Saturn --type arch --version v1 --workload densenet161 --layer_id 0 --batch_size 1 --expanded_scope 0.2 --all_DNN --weight_num 10
# echo "-----------------------------------------------5"
# python main.py --optim_obj EDP --population 20 --epochs 100  --accelerator Simba --mapper Saturn --type arch --version v1 --workload densenet161 --layer_id 0 --batch_size 1 --expanded_scope 0.2 --all_DNN --weight_num 5
# echo "-----------------------------------------------1"
# python main.py --optim_obj EDP --population 20 --epochs 100  --accelerator Simba --mapper Saturn --type arch --version v1 --workload densenet161 --layer_id 0 --batch_size 1 --expanded_scope 0.2 --all_DNN --weight_num 1
# echo "-----------------------------------------------50 --all_workload"
# python main.py --optim_obj EDP --population 20 --epochs 100  --accelerator Simba --mapper Saturn --type arch --version v1 --workload densenet161 --layer_id 0 --batch_size 1 --expanded_scope 0.2 --all_DNN --weight_num 50 --all_workload
# echo "-----------------------------------------------40 --all_workload"
# python main.py --optim_obj EDP --population 20 --epochs 100  --accelerator Simba --mapper Saturn --type arch --version v1 --workload densenet161 --layer_id 0 --batch_size 1 --expanded_scope 0.2 --all_DNN --weight_num 40 --all_workload 
# echo "-----------------------------------------------30 --all_workload"
# python main.py --optim_obj EDP --population 20 --epochs 100  --accelerator Simba --mapper Saturn --type arch --version v1 --workload densenet161 --layer_id 0 --batch_size 1 --expanded_scope 0.2 --all_DNN --weight_num 30 --all_workload 
# echo "-----------------------------------------------20 --all_workload"
# python main.py --optim_obj EDP --population 20 --epochs 100  --accelerator Simba --mapper Saturn --type arch --version v1 --workload densenet161 --layer_id 0 --batch_size 1 --expanded_scope 0.2 --all_DNN --weight_num 20 --all_workload
# echo "-----------------------------------------------10 --all_workload"
# python main.py --optim_obj EDP --population 20 --epochs 100  --accelerator Simba --mapper Saturn --type arch --version v1 --workload densenet161 --layer_id 0 --batch_size 1 --expanded_scope 0.2 --all_DNN --weight_num 10 --all_workload
# echo "-----------------------------------------------5 --all_workload"
# python main.py --optim_obj EDP --population 20 --epochs 100  --accelerator Simba --mapper Saturn --type arch --version v1 --workload densenet161 --layer_id 0 --batch_size 1 --expanded_scope 0.2 --all_DNN --weight_num 5 --all_workload
# python main.py --optim_obj EDP --population 20 --epochs 200  --accelerator Simba --mapper Saturn --type arch --version v1 --workload llama2_7B --layer_id 0 --batch_size 1 --expanded_scope 0.2 --all_DNN --weight_num 6
# python main.py --optim_obj EDP --population 20 --epochs 200  --accelerator Simba --mapper Saturn --type arch --version v1 --workload llama2_13B --layer_id 0 --batch_size 1 --expanded_scope 0.2 --all_DNN --weight_num 6
# python main.py --optim_obj EDP --population 20 --epochs 20  --accelerator Simba --mapper Saturn --type arch --version v1 --workload llama2_34B --layer_id 0 --batch_size 1 --expanded_scope 0.2 --all_DNN --weight_num 4
# python main.py --optim_obj EDP --population 20 --epochs 20  --accelerator Gemmini --mapper Saturn --type arch --version v1 --workload mm --layer_id 0 --batch_size 1 --expanded_scope 0.2 --all_DNN --weight_num 4
# python main.py --optim_obj EDP --population 20 --epochs 20  --accelerator NEW --mapper Saturn --type arch --version v1 --workload llama2_7B --layer_id 0 --batch_size 1 --expanded_scope 0.2 --all_DNN --weight_num 6
echo "-----------------------------------------------1 --all_workload"
# python main.py --optim_obj EDP --population 20 --epochs 100  --accelerator Gemmini --mapper Random --type arch --version v1 --workload mm --layer_id 0 --batch_size 1 --expanded_scope 0.2 --all_DNN --weight_num 1
# python main.py --optim_obj EDP --population 20 --epochs 100  --accelerator Gemmini --mapper BO --type arch --version v1 --workload mm --layer_id 0 --batch_size 1 --expanded_scope 0.2 --all_DNN --weight_num 1
# python main.py --optim_obj EDP --population 20 --epochs 20  --accelerator Gemmini --mapper Saturn --type arch --version v1 --workload vgg16 --layer_id 0 --batch_size 1 --expanded_scope 0.2 --all_DNN --weight_num 1 
# python main.py --optim_obj EDP --population 20 --epochs 20  --accelerator Gemmini --mapper Saturn --type arch --version v1 --workload unet --layer_id 0 --batch_size 1 --expanded_scope 0.2 --all_DNN --weight_num 1 
# python main.py --optim_obj EDP --population 20 --epochs 20  --accelerator Gemmini --mapper Saturn --type arch --version v1 --workload resnet50 --layer_id 0 --batch_size 1 --expanded_scope 0.2 --all_DNN --weight_num 1 
# python main.py --optim_obj EDP --population 20 --epochs 20  --accelerator Gemmini --mapper Saturn --type arch --version v1 --workload retinanet --layer_id 0 --batch_size 1 --expanded_scope 0.2 --all_DNN --weight_num 1 
# python main.py --optim_obj EDP --population 20 --epochs 20  --accelerator Gemmini --mapper Saturn --type arch --version v1 --workload bert --layer_id 0 --batch_size 1 --expanded_scope 0.2 --all_DNN --weight_num 1 
# python main.py --optim_obj EDP --population 20 --epochs 20  --accelerator Gemmini --mapper Saturn --type arch --version v1 --workload llama2_7B --layer_id 0 --batch_size 1 --expanded_scope 0.2 --all_DNN --weight_num 1 
# python main.py --optim_obj EDP --population 20 --epochs 20  --accelerator Gemmini --mapper Saturn --type arch --version v1 --workload llama2_13B --layer_id 0 --batch_size 1 --expanded_scope 0.2 --all_DNN --weight_num 1 
# python main.py --optim_obj EDP --population 20 --epochs 20  --accelerator Gemmini --mapper Saturn --type arch --version v1 --workload llama2_34B --layer_id 0 --batch_size 1 --expanded_scope 0.2 --all_DNN --weight_num 1 
cd ..