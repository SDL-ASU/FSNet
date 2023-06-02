python my_slurm_manager.py -n 1080Ti_special -gpu 4 -i 100 -script slurm_script_resnet_fm18_cifar.sh -log './logs/resnet_fm18_compression1x1_4.log' -on_reserved 0

python my_slurm_manager.py -n 1080Ti -gpu 8 -i 100 -script slurm_script_resnet_fm18_compression1x1_8_imagenet.sh  -log './logs/resnet_fm18_imagenet_compression1x1_8.log' -on_reserved 0

python my_slurm_manager.py -n 1080Ti -gpu 8 -i 100 -script slurm_script_resnet_fm50_compression1x1_8_imagenet.sh  -log './logs/resnet_fm50_imagenet_compression1x1_8.log' -on_reserved 0 

python my_slurm_manager.py -n 1080Ti -gpu 8 -i 100 -script slurm_script_resnet18_imagenet.sh -log './logs/resnet18_imagenet.log' -on_reserved 0 


python my_slurm_manager.py -gpu 4 -i 100 -script slurm_script_resnet_fm18_compression1x1_8_cifar.sh -log './logs/resnet_fm18_compression1x1_8.log' -on_reserved 1 -reservation world -nodelist asimov-175

python my_slurm_manager.py -gpu 4 -i 100 -script slurm_script_resnet_fm34_cifar.sh -log './logs/resnet_fm34_compression1x1_4.log' -on_reserved 1 -reservation world -nodelist asimov-178

python my_slurm_manager.py -gpu 4 -i 100 -script slurm_script_resnet_fm34_compression1x1_8_cifar.sh -log './logs/resnet_fm34_compression1x1_8.log' -on_reserved 1 -reservation world -nodelist asimov-176

python my_slurm_manager.py -gpu 4 -i 100 -script slurm_script_resnet_fm50_cifar.sh -log './logs/resnet_fm50_compression1x1_4.log' -on_reserved 1 -reservation world -nodelist asimov-176

python my_slurm_manager.py -gpu 4 -i 100 -script slurm_script_resnet_fm50_compression1x1_8_cifar.sh -log './logs/resnet_fm50_compression1x1_8.log' -on_reserved 1 -reservation world -nodelist asimov-178

python my_slurm_manager.py -gpu 4 -i 100 -script slurm_script_densenet_fm121_cifar.sh -log './logs/densenet_fm121_compression1x1_4.log' -on_reserved 1 -reservation world -nodelist asimov-179

python my_slurm_manager.py -gpu 4 -i 100 -script slurm_script_densenet_fm121_compression1x1_8_cifar.sh -log './logs/densenet_fm121_compression1x1_8.log' -on_reserved 1 -reservation world -nodelist asimov-180

