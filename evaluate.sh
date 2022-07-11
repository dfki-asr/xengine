mkdir -p build
cd build

# output directory
machine="iris"
#machine="gen9"
outdir=EVALUATION/evaluation_${machine}
mkdir -p ${outdir}

# what devices should we evaluate?
device_file=../data/devices/devices_auto.txt
#device_file=../data/devices/devices_cpu.txt
#device_file=../data/devices/devices_gpu.txt
#device_file=../data/devices/devices_cpu_gpu_gpu.txt

# insert batchsizes that should be evaluated
lenet_bs=(64)
vgg16_bs=()
vgg19_bs=()
resnet18_bs=()
resnet34_bs=()
resnet50_bs=()
googlenet_bs=()
unet2D_bs=()
unet3D_bs=()

# Number of cores for experiments
num_cpu_cores=(24)

# mode: inference or training
#modi=("inf" "train")
#mode="train"
#modeIdx=1
mode="inf"
modeIdx=0


#for modeIdx in ${!modi[@]}; do
  #mode=${modi[${modeIdx}]}
  for ncores in ${num_cpu_cores[@]}; do
    outdir_ncores="${outdir}/${ncores}_cores"
    mkdir -p ${outdir_ncores}
    for bs in ${lenet_bs[@]}; do
      echo "lenet ${mode} ${bs}"
      cmd="./examples/xengine lenet ${bs} ${modeIdx} ${outdir_ncores} ${device_file} > ${outdir_ncores}/lenet_${bs}_${mode}_marvellous.log"
      echo $cmd
      $cmd
    done
    for bs in ${vgg16_bs[@]}; do
      echo "vgg16 ${mode} ${bs}"
      cmd="./examples/xengine vgg16 ${bs} ${modeIdx} ${outdir_ncores} ${device_file} > ${outdir_ncores}/vgg16_${bs}_${mode}_marvellous.log"
      echo $cmd
      $cmd
    done
    for bs in ${vgg19_bs[@]}; do
      echo "vgg19 ${mode} ${bs}"
      cmd="./examples/xengine vgg19 ${bs} ${modeIdx} ${outdir_ncores} ${device_file} > ${outdir_ncores}/vgg19_${bs}_${mode}_marvellous.log"
      echo $cmd
      $cmd
    done
    for bs in ${resnet18_bs[@]}; do
      echo "resnet18 ${mode} ${bs}"
      cmd="./examples/xengine resnet18 ${bs} ${modeIdx} ${outdir_ncores} ${device_file} > ${outdir_ncores}/resnet18_${bs}_${mode}_marvellous.log"
      echo $cmd
      $cmd
    done
    for bs in ${resnet34_bs[@]}; do
      echo "resnet34 ${mode} ${bs}"
      cmd="./examples/xengine resnet34 ${bs} ${modeIdx} ${outdir_ncores} ${device_file} > ${outdir_ncores}/resnet34_${bs}_${mode}_marvellous.log"
      echo $cmd
      $cmd
    done
    for bs in ${resnet50_bs[@]}; do
      echo "resnet50 ${mode} ${bs}"
      cmd="./examples/xengine resnet50 ${bs} ${modeIdx} ${outdir_ncores} ${device_file} > ${outdir_ncores}/resnet50_${bs}_${mode}_marvellous.log"
      echo $cmd
      $cmd
    done
    for bs in ${googlenet_bs[@]}; do
      echo "googlenet ${mode} ${bs}"
      cmd="./examples/xengine googlenet ${bs} ${modeIdx} ${outdir_ncores} ${device_file} > ${outdir_ncores}/googlenet_${bs}_${mode}_marvellous.log"
      echo $cmd
      $cmd
    done
    for bs in ${unet2D_bs[@]}; do
      echo "unet2D ${mode} ${bs}"
      cmd="./examples/xengine unet2D ${bs} ${modeIdx} ${outdir_ncores} ${device_file} > ${outdir_ncores}/unet2D_${bs}_${mode}_marvellous.log"
      echo $cmd
      $cmd
    done
    for bs in ${unet3D_bs[@]}; do
      echo "unet3D ${mode} ${bs}"
      cmd="./examples/xengine unet3D ${bs} ${modeIdx} ${outdir_ncores} ${device_file} > ${outdir_ncores}/unet3D_${bs}_${mode}_marvellous.log"
      echo $cmd
      $cmd
    done
  done
#done
