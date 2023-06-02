srun -p 1080Ti_special --gres=gpu:4 --cpus-per-task 2 -n 1 --pty bash -l

srun -p 1080Ti --gres=gpu:1 --cpus-per-task 2 -n 1 --pty bash -l

scontrol show res

sbatch --reservation=world --nodelist=asimov-[174-176],asimov-178 slurm_script.sh

python my_slurm_manager

python quantization.py --checkpoint_folder_fm saved_models/checkpoint_resnet_fm18_compression1x1_8 --evaluate --compression1x1 8 --ori_arch resnet18 --fm_arch resnet_fm18

python quantization.py --checkpoint_folder_ori saved_models/checkpoint_resnet34 --checkpoint_folder_fm saved_models/checkpoint_resnet_fm34_compression1x1_8 --evaluate

python quantization.py --checkpoint_folder_ori saved_models/checkpoint_resnet50 --checkpoint_folder_fm saved_models/checkpoint_resnet_fm50_compression1x1_8 --evaluate

python quantization.py --checkpoint_folder_ori saved_models/checkpoint_resnet101 --checkpoint_folder_fm saved_models/checkpoint_resnet_fm101_compression1x1_8 --evaluate

python quantization.py --checkpoint_folder_ori saved_models/checkpoint_densenet121 --checkpoint_folder_fm saved_models/checkpoint_densenet_fm121_compression1x1_8 --evaluate

#command for quantization.py for imagenet
python quantization.py --ori_arch resnet50_imagenet --fm_arch resnet_fm50_imagenet --checkpoint_folder_ori saved_models/checkpoint_resnet50_imagenet --checkpoint_folder_fm saved_models/checkpoint_resnet_fm50_imagenet_compression1x1_8 --dataset imagenet --compression1x1 8 --evaluate

python quantization_baseline.py --ori_arch resnet50_imagenet --fm_arch resnet_fm50_imagenet --checkpoint_folder_fm saved_models/checkpoint_resnet_fm50_imagenet_compression_2 --dataset imagenet --compression_ratio 2 --evaluate

python my_slurm_submitter.py -n 1080Ti -c 2 -g 4 -i 100 -f slurm_commands_cifar.txt -s scripts_cifar

python my_slurm_submitter.py -n 1080Ti -c 2 -g 4 -i 100 -f slurm_commands_imagenet.txt -s scripts_imagenet
