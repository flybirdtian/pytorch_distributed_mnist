# pytorch_distributed_mnist
The distributed training code for mnist example using pytorch 

# Environments requirements:
1. Need to install torch and torchvision in python environments
2. This fits for a single node with multiple GPUs, you need at least one NVIDIA GPU device

# Traning:
There are two ways for traning, which is very similar:
1. Way 1: use torch.distributed.launch to launch multiple processes
   * uncomment the following line near the end of file multi_proc_single_gpu.py

     > run_dist_launch(args)
     
     and comment the last line:
     
     > demo_spawn(ngpus, args)
   * run command in a console:
     > CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 multi_proc_single_gpu.py --world-size 4 --workers 4
      
     remember to modify 'nproc_per_node', 'world-size' as well as 'workers' according to the number of GPUs in CUDA_VISIBLE_DEVICES, 'workers' can be n*world-size

2. Way 2: use torch.multiprocessing.spawn to generate multiple processes
   * uncomment the following line at the end of file multi_proc_single_gpu.py

     > demo_spawn(ngpus, args)
     
     and comment the line:
     
     > run_dist_launch(args)

   * run command in a console:
     > CUDA_VISIBLE_DEVICES=0,1,2,3 python multi_proc_single_gpu.py --world-size 4
    
     remember to modify 'world-size' as well as 'workers' according to the number of GPUs in CUDA_VISIBLE_DEVICES, 'workers' can be n*world-size

# Pre-Download dataset  
You need to download Mnist dataset before running and put it to 'data' folder in the sub-directory of this project 
  
   <em>or</em>
   
   if use way 1: you can simply run the following until download finished:
   > CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 multi_proc_single_gpu.py --world-size 1
   
   <em>or</em>

   if use way 2: you can simply run the following command until downlaod finished:
   > CUDA_VISIBLE_DEVICES=0 python multi_proc_single_gpu.py --world-size 1

# Resume training:
  add resume arguments based on traning command:
  --resume [path-to-checkpoint]

# Evaluate:
   In evaluate, you only need to test on a single GPU
   if use way 1, run:
   > CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 multi_proc_single_gpu.py --world-size 1 --evaluate --resume [path-to-checkpoint]

   if use way 2, run:
   > CUDA_VISIBLE_DEVICES=0 python multi_proc_single_gpu.py --world-size 1 --evaluate --resume [path-to-checkpoint]


