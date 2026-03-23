# cd ./src
# start_time=$(date +%s.%N)  # 获取开始时间（纳秒级时间戳，如：1689023456.123456789）
# for layer_id in {56..56}
# do
#     printf "网络: deepbench, 层数: %d\n" $layer_id 
#     python main.py --optim_obj latency --population 50 --epochs 50 --accelerator Simba --mapper MARS --type arch --version v1 --workload deepbench --layer_id $layer_id --batch_size 1 --expanded_scope 0.1
# done
# end_time=$(date +%s.%N)    # 获取结束时间
# elapsed_time=$(echo "scale=3; $end_time - $start_time" | bc)  # 计算差值，保留3位小数（秒）

# echo "执行时间：${elapsed_time} 秒"

# start_time=$(date +%s.%N)  # 获取开始时间（纳秒级时间戳，如：1689023456.123456789）

cd ./src
# start_time=$(date +%s.%N)  # 获取开始时间（纳秒级时间戳，如：1689023456.123456789）
# for layer_id in {47..47}
# do
#     printf "网络: deepbench, 层数: %d\n" $layer_id 
#     python main.py --optim_obj energy --population 80 --epochs 80 --accelerator Simba --mapper MARS --type arch --version v1 --workload deepbench --layer_id $layer_id --batch_size 1 --expanded_scope 0.1
# done
# end_time=$(date +%s.%N)    # 获取结束时间
# elapsed_time=$(echo "scale=3; $end_time - $start_time" | bc)  # 计算差值，保留3位小数（秒）

# echo "执行时间：${elapsed_time} 秒"

# start_time=$(date +%s.%N)  # 获取开始时间（纳秒级时间戳，如：1689023456.123456789）
# for layer_id in {75..75}
# do
#     printf "网络: deepbench, 层数: %d\n" $layer_id 
#     python main.py --optim_obj all --population 100 --epochs 100 --accelerator Simba --mapper MARS --type arch --version v1 --workload deepbench --layer_id $layer_id --batch_size 1 --expanded_scope 0.1
    # python main.py --optim_obj latency --population 100 --epochs 100 --accelerator Simba --mapper MARS --type arch --version v1 --workload deepbench --layer_id $layer_id --batch_size 1 --expanded_scope 0.1
    # python main.py --optim_obj energy --population 100 --epochs 100 --accelerator Simba --mapper MARS --type arch --version v1 --workload deepbench --layer_id $layer_id --batch_size 1 --expanded_scope 0.1
#     python main.py --optim_obj EDP --population 100 --epochs 100 --accelerator Simba --mapper MARS --type arch --version v1 --workload deepbench --layer_id $layer_id --batch_size 1 --expanded_scope 0.1
# done
# end_time=$(date +%s.%N)    # 获取结束时间
# elapsed_time=$(echo "scale=3; $end_time - $start_time" | bc)  # 计算差值，保留3位小数（秒）

# echo "执行时间：${elapsed_time} 秒"

# for layer_id in {182..182}
# do
#     printf "网络: conv, 层数: %d\n" $layer_id 
#     python main.py --optim_obj latency --population 50 --epochs 50 --accelerator Simba --mapper MARS --type arch --version v1 --workload conv --layer_id $layer_id --batch_size 1 --expanded_scope 0.1
# done
# end_time=$(date +%s.%N)    # 获取结束时间
# elapsed_time=$(echo "scale=3; $end_time - $start_time" | bc)  # 计算差值，保留3位小数（秒）

# echo "执行时间：${elapsed_time} 秒"

# start_time=$(date +%s.%N)  # 获取开始时间（纳秒级时间戳，如：1689023456.123456789）
# for layer_id in {0..6}
# do
#     for batch_size in {1..64}
#     do
#         printf "网络: tensor, 层数: %d\n" $layer_id 
#         python main.py --optim_obj latency --population 50 --epochs 20 --accelerator Simba --mapper MARS --type arch --version v1 --workload tensor --layer_id $layer_id --batch_size $batch_size --expanded_scope 0.1
#     done
# done
# end_time=$(date +%s.%N)    # 获取结束时间
# elapsed_time=$(echo "scale=3; $end_time - $start_time" | bc)  # 计算差值，保留3位小数（秒）

# echo "执行时间：${elapsed_time} 秒"

# for layer_id in {0..20}
# do
#     printf "网络: operators, 层数: %d\n" $layer_id 
#     python main.py --optim_obj latency --population 50 --epochs 20 --accelerator TensorCore --mapper MARS --type arch --version v1 --workload operators --layer_id $layer_id --batch_size 1 --expanded_scope 0.1
# done
# for layer_id in {0..0}
# do
#     printf "网络: mobilenet, 层数: %d\n" $layer_id
#     python main.py --optim_obj energy --population 50 --epochs 50 --accelerator TensorCore --mapper MARS --type arch --version v1 --workload mobilenet --layer_id $layer_id --batch_size 1 --expanded_scope 0.1
    # python main.py --optim_obj EDP --population 50 --epochs 50 --accelerator TensorCore --mapper MARS --type arch --version v1 --workload mobilenet --layer_id $layer_id --batch_size 1 --expanded_scope 0.1
    # python main.py --optim_obj energy --population 10 --epochs 10 --accelerator Simba --mapper MARS --type arch --version v1 --workload mobilenet --layer_id $layer_id --batch_size 1 --expanded_scope 0.1
    # python main.py --optim_obj latency --population 100 --epochs 100 --accelerator Simba --mapper MARS --type arch --version v1 --workload mobilenet --layer_id $layer_id --batch_size 1 --expanded_scope 0.1
    # python main.py --optim_obj EDP --population 100 --epochs 100 --accelerator Simba --mapper MARS --type arch --version v1 --workload mobilenet --layer_id $layer_id --batch_size 1 --expanded_scope 0.1
#     python main.py --optim_obj energy --population 50 --epochs 50 --accelerator Eyeriss --mapper MARS --type arch --version v1 --workload mobilenet --layer_id $layer_id --batch_size 1 --expanded_scope 0.1
#     python main.py --optim_obj EDP --population 50 --epochs 50 --accelerator Eyeriss --mapper MARS --type arch --version v1 --workload mobilenet --layer_id $layer_id --batch_size 1 --expanded_scope 0.1
# done
# for layer_id in {0..8}
# do
#     printf "网络: transformer, 层数: %d\n" $layer_id
#     python main.py --optim_obj energy --population 100 --epochs 100 --accelerator TensorCore --mapper MARS --type arch --version v1 --workload transformer --layer_id $layer_id --batch_size 1 --expanded_scope 0.1
#     python main.py --optim_obj EDP --population 100 --epochs 100 --accelerator TensorCore --mapper MARS --type arch --version v1 --workload transformer --layer_id $layer_id --batch_size 1 --expanded_scope 0.1
#     python main.py --optim_obj latency --population 100 --epochs 100 --accelerator TensorCore --mapper MARS --type arch --version v1 --workload transformer --layer_id $layer_id --batch_size 1 --expanded_scope 0.1
#     python main.py --optim_obj energy --population 100 --epochs 100 --accelerator Simba --mapper MARS --type arch --version v1 --workload transformer --layer_id $layer_id --batch_size 1 --expanded_scope 0.1
#     python main.py --optim_obj latency --population 100 --epochs 100 --accelerator Simba --mapper MARS --type arch --version v1 --workload transformer --layer_id $layer_id --batch_size 1 --expanded_scope 0.1
#     python main.py --optim_obj EDP --population 100 --epochs 100 --accelerator Simba --mapper MARS --type arch --version v1 --workload transformer --layer_id $layer_id --batch_size 1 --expanded_scope 0.1
#     python main.py --optim_obj energy --population 100 --epochs 100 --accelerator Eyeriss --mapper MARS --type arch --version v1 --workload transformer --layer_id $layer_id --batch_size 1 --expanded_scope 0.1
#     python main.py --optim_obj EDP --population 100 --epochs 100 --accelerator Eyeriss --mapper MARS --type arch --version v1 --workload transformer --layer_id $layer_id --batch_size 1 --expanded_scope 0.1
#     python main.py --optim_obj latency --population 100 --epochs 100 --accelerator Eyeriss --mapper MARS --type arch --version v1 --workload transformer --layer_id $layer_id --batch_size 1 --expanded_scope 0.1
# done
# for layer_id in {12..12}
# do
#     printf "网络: unet, 层数: %d\n" $layer_id
#     python main.py --optim_obj energy --population 50 --epochs 50 --accelerator TensorCore --mapper MARS --type arch --version v1 --workload unet --layer_id $layer_id --batch_size 1 --expanded_scope 0.1
    # python main.py --optim_obj EDP --population 50 --epochs 50 --accelerator TensorCore --mapper MARS --type arch --version v1 --workload unet --layer_id $layer_id --batch_size 1 --expanded_scope 0.1
    # python main.py --optim_obj latency --population 50 --epochs 50 --accelerator TensorCore --mapper MARS --type arch --version v1 --workload unet --layer_id $layer_id --batch_size 1 --expanded_scope 0.1
    # python main.py --optim_obj energy --population 80 --epochs 80 --accelerator Simba --mapper MARS --type arch --version v1 --workload unet --layer_id $layer_id --batch_size 1 --expanded_scope 0.1
    # python main.py --optim_obj EDP --population 50 --epochs 50 --accelerator Simba --mapper MARS --type arch --version v1 --workload unet --layer_id $layer_id --batch_size 1 --expanded_scope 0.1
#     python main.py --optim_obj energy --population 50 --epochs 50 --accelerator Eyeriss --mapper MARS --type arch --version v1 --workload unet --layer_id $layer_id --batch_size 1 --expanded_scope 0.1
#     python main.py --optim_obj EDP --population 50 --epochs 50 --accelerator Eyeriss --mapper MARS --type arch --version v1 --workload unet --layer_id $layer_id --batch_size 1 --expanded_scope 0.1
#     python main.py --optim_obj latency --population 50 --epochs 50 --accelerator Eyeriss --mapper MARS --type arch --version v1 --workload unet --layer_id $layer_id --batch_size 1 --expanded_scope 0.1
# done
# for layer_id in {14..14}
# do
#     printf "网络: resnet50, 层数: %d\n" $layer_id 
#     python main.py --optim_obj energy --population 50 --epochs 50 --accelerator TensorCore --mapper MARS --type arch --version v1 --workload resnet50 --layer_id $layer_id --batch_size 1 --expanded_scope 0.1
    # python main.py --optim_obj EDP --population 50 --epochs 50 --accelerator TensorCore --mapper MARS --type arch --version v1 --workload resnet50 --layer_id $layer_id --batch_size 1 --expanded_scope 0.1
    # python main.py --optim_obj latency --population 50 --epochs 50 --accelerator TensorCore --mapper MARS --type arch --version v1 --workload resnet50 --layer_id $layer_id --batch_size 1 --expanded_scope 0.1
    # python main.py --optim_obj energy --population 50 --epochs 50 --accelerator Simba --mapper MARS --type arch --version v1 --workload resnet50 --layer_id $layer_id --batch_size 1 --expanded_scope 0.1
    # python main.py --optim_obj EDP --population 50 --epochs 80 --accelerator Simba --mapper MARS --type arch --version v1 --workload resnet50 --layer_id $layer_id --batch_size 1 --expanded_scope 0.1
#     python main.py --optim_obj energy --population 50 --epochs 50 --accelerator Eyeriss --mapper MARS --type arch --version v1 --workload resnet50 --layer_id $layer_id --batch_size 1 --expanded_scope 0.1
#     python main.py --optim_obj EDP --population 50 --epochs 50 --accelerator Eyeriss --mapper MARS --type arch --version v1 --workload resnet50 --layer_id $layer_id --batch_size 1 --expanded_scope 0.1
#     python main.py --optim_obj latency --population 50 --epochs 50 --accelerator Eyeriss --mapper MARS --type arch --version v1 --workload resnet50 --layer_id $layer_id --batch_size 1 --expanded_scope 0.1
# done

# for layer_id in {2..3}
# do
#     printf "网络: GCN_Citeseer, 层数: %d\n" $layer_id 
#     python main.py --optim_obj energy --population 100 --epochs 100 --accelerator TensorCore --mapper MARS --type arch --version v1 --workload GCN_Citeseer --layer_id $layer_id --batch_size 1 --expanded_scope 0.1
#     python main.py --optim_obj EDP --population 100 --epochs 100 --accelerator TensorCore --mapper MARS --type arch --version v1 --workload GCN_Citeseer --layer_id $layer_id --batch_size 1 --expanded_scope 0.1
#     python main.py --optim_obj latency --population 100 --epochs 100 --accelerator TensorCore --mapper MARS --type arch --version v1 --workload GCN_Citeseer --layer_id $layer_id --batch_size 1 --expanded_scope 0.1
#     python main.py --optim_obj energy --population 100 --epochs 100 --accelerator Simba --mapper MARS --type arch --version v1 --workload GCN_Citeseer --layer_id $layer_id --batch_size 1 --expanded_scope 0.1
#     python main.py --optim_obj EDP --population 100 --epochs 100 --accelerator Simba --mapper MARS --type arch --version v1 --workload GCN_Citeseer --layer_id $layer_id --batch_size 1 --expanded_scope 0.1
#     python main.py --optim_obj energy --population 100 --epochs 100 --accelerator Eyeriss --mapper MARS --type arch --version v1 --workload GCN_Citeseer --layer_id $layer_id --batch_size 1 --expanded_scope 0.1
#     python main.py --optim_obj EDP --population 100 --epochs 100 --accelerator Eyeriss --mapper MARS --type arch --version v1 --workload GCN_Citeseer --layer_id $layer_id --batch_size 1 --expanded_scope 0.1
#     python main.py --optim_obj latency --population 100 --epochs 100 --accelerator Eyeriss --mapper MARS --type arch --version v1 --workload GCN_Citeseer --layer_id $layer_id --batch_size 1 --expanded_scope 0.1
# done

# for layer_id in {0..3}
# do
#     printf "网络: GCN_Citeseer, 层数: %d\n" $layer_id 
#     python main.py --optim_obj latency --population 100 --epochs 100 --accelerator Simba --mapper MARS --type arch --version v1 --workload GCN_Citeseer --layer_id $layer_id --batch_size 1 --expanded_scope 0.1
# done

# for layer_id in {0..0}
# do
#     printf "网络: GCN_reddit, 层数: %d\n" $layer_id 
#     python main.py --optim_obj latency --population 100 --epochs 100 --accelerator Simba --mapper MARS --type arch --version v1 --workload GCN_reddit --layer_id $layer_id --batch_size 1 --expanded_scope 0.1
# done

# for layer_id in {1..3}
# do
#     printf "网络: GCN_reddit, 层数: %d\n" $layer_id 
#     python main.py --optim_obj energy --population 100 --epochs 100 --accelerator TensorCore --mapper MARS --type arch --version v1 --workload GCN_reddit --layer_id $layer_id --batch_size 1 --expanded_scope 0.1
#     python main.py --optim_obj EDP --population 100 --epochs 100 --accelerator TensorCore --mapper MARS --type arch --version v1 --workload GCN_reddit --layer_id $layer_id --batch_size 1 --expanded_scope 0.1
#     python main.py --optim_obj latency --population 100 --epochs 100 --accelerator TensorCore --mapper MARS --type arch --version v1 --workload GCN_reddit --layer_id $layer_id --batch_size 1 --expanded_scope 0.1
#     python main.py --optim_obj energy --population 100 --epochs 100 --accelerator Simba --mapper MARS --type arch --version v1 --workload GCN_reddit --layer_id $layer_id --batch_size 1 --expanded_scope 0.1
#     python main.py --optim_obj EDP --population 100 --epochs 100 --accelerator Simba --mapper MARS --type arch --version v1 --workload GCN_reddit --layer_id $layer_id --batch_size 1 --expanded_scope 0.1
#     python main.py --optim_obj latency --population 100 --epochs 100 --accelerator Simba --mapper MARS --type arch --version v1 --workload GCN_reddit --layer_id $layer_id --batch_size 1 --expanded_scope 0.1
#     python main.py --optim_obj energy --population 100 --epochs 100 --accelerator Eyeriss --mapper MARS --type arch --version v1 --workload GCN_reddit --layer_id $layer_id --batch_size 1 --expanded_scope 0.1
#     python main.py --optim_obj EDP --population 100 --epochs 100 --accelerator Eyeriss --mapper MARS --type arch --version v1 --workload GCN_reddit --layer_id $layer_id --batch_size 1 --expanded_scope 0.1
#     python main.py --optim_obj latency --population 100 --epochs 100 --accelerator Eyeriss --mapper MARS --type arch --version v1 --workload GCN_reddit --layer_id $layer_id --batch_size 1 --expanded_scope 0.1
# done

# for layer_id in {0..0}
# do
#     printf "网络: resnet50, 层数: %d\n" $layer_id 
#     python main.py --optim_obj EDP --population 100 --epochs 20 --accelerator Gemmini --mapper MARS --type arch --version v1 --workload resnet50 --layer_id $layer_id --batch_size 1 --expanded_scope 0.0 --solver lp
# done
for layer_id in {5..5}
do
    printf "网络: alexnet, 层数: %d\n" $layer_id 
    python main.py --optim_obj EDP --population 100 --epochs 20 --accelerator Gemmini --mapper MARS --type arch --version v1 --workload alexnet --layer_id $layer_id --batch_size 1 --expanded_scope 0.0 --solver lp
done
# for layer_id in {0..5}
# do
#     printf "网络: bert, 层数: %d\n" $layer_id 
#     python main.py --optim_obj EDP --population 50 --epochs 50 --accelerator Gemmini --mapper MARS --type arch --version v1 --workload bert --layer_id $layer_id --batch_size 1 --expanded_scope 0.0 --solver lp
# done
# for layer_id in {0..22}
# do
#     printf "网络: unet, 层数: %d\n" $layer_id 
#     python main.py --optim_obj EDP --population 100 --epochs 100 --accelerator Gemmini --mapper MARS --type arch --version v1 --workload unet --layer_id $layer_id --batch_size 1 --expanded_scope 0.0 --solver lp
# done

# for layer_id in {0..5}
# do
#     printf "网络: gpt3, 层数: %d\n" $layer_id 
#     python main.py --optim_obj energy --population 100 --epochs 100 --accelerator TensorCore --mapper MARS --type arch --version v1 --workload gpt3 --layer_id $layer_id --batch_size 1 --expanded_scope 0.1
#     python main.py --optim_obj EDP --population 100 --epochs 100 --accelerator TensorCore --mapper MARS --type arch --version v1 --workload gpt3 --layer_id $layer_id --batch_size 1 --expanded_scope 0.1
#     python main.py --optim_obj latency --population 100 --epochs 100 --accelerator TensorCore --mapper MARS --type arch --version v1 --workload gpt3 --layer_id $layer_id --batch_size 1 --expanded_scope 0.1
#     python main.py --optim_obj energy --population 100 --epochs 100 --accelerator Simba --mapper MARS --type arch --version v1 --workload gpt3 --layer_id $layer_id --batch_size 1 --expanded_scope 0.1
#     python main.py --optim_obj EDP --population 100 --epochs 100 --accelerator Simba --mapper MARS --type arch --version v1 --workload gpt3 --layer_id $layer_id --batch_size 1 --expanded_scope 0.1
#     python main.py --optim_obj latency --population 100 --epochs 100 --accelerator Simba --mapper MARS --type arch --version v1 --workload gpt3 --layer_id $layer_id --batch_size 1 --expanded_scope 0.1
#     python main.py --optim_obj energy --population 100 --epochs 100 --accelerator Eyeriss --mapper MARS --type arch --version v1 --workload gpt3 --layer_id $layer_id --batch_size 1 --expanded_scope 0.1
#     python main.py --optim_obj EDP --population 100 --epochs 100 --accelerator Eyeriss --mapper MARS --type arch --version v1 --workload gpt3 --layer_id $layer_id --batch_size 1 --expanded_scope 0.1
#     python main.py --optim_obj latency --population 100 --epochs 100 --accelerator Eyeriss --mapper MARS --type arch --version v1 --workload gpt3 --layer_id $layer_id --batch_size 1 --expanded_scope 0.1
# done

# for layer_id in {0..75}
# do
#     printf "网络: deepbench, 层数: %d\n" $layer_id 
#     python main.py --optim_obj energy --population 100 --epochs 100 --accelerator TensorCore --mapper MARS --type arch --version v1 --workload deepbench --layer_id $layer_id --batch_size 1 --expanded_scope 0.1
#     python main.py --optim_obj EDP --population 100 --epochs 100 --accelerator TensorCore --mapper MARS --type arch --version v1 --workload deepbench --layer_id $layer_id --batch_size 1 --expanded_scope 0.1
#     python main.py --optim_obj latency --population 100 --epochs 100 --accelerator TensorCore --mapper MARS --type arch --version v1 --workload deepbench --layer_id $layer_id --batch_size 1 --expanded_scope 0.1
#     python main.py --optim_obj energy --population 100 --epochs 100 --accelerator Eyeriss --mapper MARS --type arch --version v1 --workload deepbench --layer_id $layer_id --batch_size 1 --expanded_scope 0.1
#     python main.py --optim_obj EDP --population 100 --epochs 100 --accelerator Eyeriss --mapper MARS --type arch --version v1 --workload deepbench --layer_id $layer_id --batch_size 1 --expanded_scope 0.1
#     python main.py --optim_obj latency --population 100 --epochs 100 --accelerator Eyeriss --mapper MARS --type arch --version v1 --workload deepbench --layer_id $layer_id --batch_size 1 --expanded_scope 0.1
# done

# for layer_id in {0..3}
# do
#     printf "网络: gpt1, 层数: %d\n" $layer_id 
#     python main.py --optim_obj energy --population 50 --epochs 50 --accelerator TensorCore --mapper MARS --type arch --version v1 --workload gpt1 --layer_id $layer_id --batch_size 1 --expanded_scope 0.1
#     python main.py --optim_obj EDP --population 50 --epochs 50 --accelerator TensorCore --mapper MARS --type arch --version v1 --workload gpt1 --layer_id $layer_id --batch_size 1 --expanded_scope 0.1
    # python main.py --optim_obj energy --population 100 --epochs 100 --accelerator Simba --mapper MARS --type arch --version v1 --workload gpt1 --layer_id $layer_id --batch_size 1 --expanded_scope 0.1
    # python main.py --optim_obj EDP --population 100 --epochs 100 --accelerator Simba --mapper MARS --type arch --version v1 --workload gpt1 --layer_id $layer_id --batch_size 1 --expanded_scope 0.1
#     python main.py --optim_obj energy --population 50 --epochs 50 --accelerator Eyeriss --mapper MARS --type arch --version v1 --workload gpt1 --layer_id $layer_id --batch_size 1 --expanded_scope 0.1
#     python main.py --optim_obj EDP --population 50 --epochs 50 --accelerator Eyeriss --mapper MARS --type arch --version v1 --workload gpt1 --layer_id $layer_id --batch_size 1 --expanded_scope 0.1
# done
# for layer_id in {5..5}
# do
#     printf "网络: alexnet, 层数: %d\n" $layer_id 
    # python main.py --optim_obj energy --population 50 --epochs 50 --accelerator TensorCore --mapper MARS --type arch --version v1 --workload alexnet --layer_id $layer_id --batch_size 1 --expanded_scope 0.1
    # python main.py --optim_obj EDP --population 50 --epochs 50 --accelerator TensorCore --mapper MARS --type arch --version v1 --workload alexnet --layer_id $layer_id --batch_size 1 --expanded_scope 0.1
    # python main.py --optim_obj energy --population 50 --epochs 100 --accelerator Simba --mapper MARS --type arch --version v1 --workload alexnet --layer_id $layer_id --batch_size 1 --expanded_scope 0.1
    # python main.py --optim_obj EDP --population 50 --epochs 80 --accelerator Simba --mapper MARS --type arch --version v1 --workload alexnet --layer_id $layer_id --batch_size 1 --expanded_scope 0.1
    # python main.py --optim_obj energy --population 50 --epochs 50 --accelerator Eyeriss --mapper MARS --type arch --version v1 --workload alexnet --layer_id $layer_id --batch_size 1 --expanded_scope 0.1
#     python main.py --optim_obj EDP --population 50 --epochs 50 --accelerator Eyeriss --mapper MARS --type arch --version v1 --workload alexnet --layer_id $layer_id --batch_size 1 --expanded_scope 0.1
# done
# for layer_id in {3..3}
# do
#     printf "网络: bertlarge, 层数: %d\n" $layer_id 
#     python main.py --optim_obj energy --population 50 --epochs 50 --accelerator TensorCore --mapper MARS --type arch --version v1 --workload bertlarge --layer_id $layer_id --batch_size 1 --expanded_scope 0.1
#     python main.py --optim_obj EDP --population 50 --epochs 50 --accelerator TensorCore --mapper MARS --type arch --version v1 --workload bertlarge --layer_id $layer_id --batch_size 1 --expanded_scope 0.1
#     python main.py --optim_obj energy --population 50 --epochs 50 --accelerator Simba --mapper MARS --type arch --version v1 --workload bertlarge --layer_id $layer_id --batch_size 1 --expanded_scope 0.1
    # python main.py --optim_obj EDP --population 50 --epochs 50 --accelerator Simba --mapper MARS --type arch --version v1 --workload bertlarge --layer_id $layer_id --batch_size 1 --expanded_scope 0.1
#     python main.py --optim_obj energy --population 50 --epochs 50 --accelerator Eyeriss --mapper MARS --type arch --version v1 --workload bertlarge --layer_id $layer_id --batch_size 1 --expanded_scope 0.1
#     python main.py --optim_obj EDP --population 50 --epochs 50 --accelerator Eyeriss --mapper MARS --type arch --version v1 --workload bertlarge --layer_id $layer_id --batch_size 1 --expanded_scope 0.1
# done

# for layer_id in {0..75}
# do
#     printf "网络: deepbench, 层数: %d\n" $layer_id 
#     python main.py --optim_obj latency --population 100 --epochs 100 --accelerator Eyeriss --mapper MARS --type arch --version v1 --workload deepbench --layer_id $layer_id --batch_size 1 --expanded_scope 0.1
#     python main.py --optim_obj energy --population 100 --epochs 100 --accelerator Eyeriss --mapper MARS --type arch --version v1 --workload deepbench --layer_id $layer_id --batch_size 1 --expanded_scope 0.1
#     python main.py --optim_obj EDP --population 100 --epochs 100 --accelerator Eyeriss --mapper MARS --type arch --version v1 --workload deepbench --layer_id $layer_id --batch_size 1 --expanded_scope 0.1
#     python main.py --optim_obj latency --population 100 --epochs 100 --accelerator TensorCore --mapper MARS --type arch --version v1 --workload deepbench --layer_id $layer_id --batch_size 1 --expanded_scope 0.1
#     python main.py --optim_obj energy --population 100 --epochs 100 --accelerator TensorCore --mapper MARS --type arch --version v1 --workload deepbench --layer_id $layer_id --batch_size 1 --expanded_scope 0.1
#     python main.py --optim_obj EDP --population 100 --epochs 100 --accelerator TensorCore --mapper MARS --type arch --version v1 --workload deepbench --layer_id $layer_id --batch_size 1 --expanded_scope 0.1
# done
cd ..