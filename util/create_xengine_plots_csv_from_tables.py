#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np


def plot(info, net, modi, N, threads, CPU, GPU, MILP):
    # Creates pyplots including the numbers of the tables in the evaluation part
    platform = info[0]
    model = info[1]
    labels_settings = []
    for i, n in enumerate(N):
        labels_settings.append(net[i] + ',\n' + modi[i] + ',\nN=' + str(n) +
                               ',\n#=' + str(threads[i]))
    labels_bars = ['CPU', 'GPU', 'MILP']
    x = np.arange(len(labels_settings))  # the label locations
    width = 0.7  # the width of the bars
    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width / 3, CPU, width / 3, label='CPU')
    rects2 = ax.bar(x, GPU, width / 3, label='GPU')
    rects3 = ax.bar(x + width / 3, MILP, width / 3, label='MILP')
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('runtime (ms)')
    ax.set_title('Runtimes for ' + model + ' on ' + platform)
    ax.set_xticks(x)
    ax.set_xticklabels(labels_settings, fontsize=5)
    ax.legend()
    fig.tight_layout()
    figure_name = platform + '_' + model + '.png'
    plt.savefig(figure_name, dpi=1200)


def toCSV(info, net, modi, N, threads, CPU, GPU, MILP):
    # Creates CSV files for the paper including the numbers of the tables in the evaluation part
    platform = info[0]
    model = info[1]
    csv_content = "x,name,CPU,GPU,MILP\n"
    for i in range(len(N)):
        name = net[i] + '\\\\' + modi[i] + '\\\\N=' + str(
            N[i]) + '\\\\\#=' + str(threads[i])
        csv_content += str(i) + ", " + name + ", " + str(CPU[i]) + ", " + str(
            GPU[i]) + ", " + str(MILP[i]) + "\n"
    print(csv_content)
    csvfile = platform + '_' + model + '.csv'
    file = open(csvfile, 'w')
    file.write(csv_content)
    file.close()


### Data from the tables in the xengine paper (evaluation part) ###

# ResNet on platform 'IRIS XE'
Iris_ResNet_info = ['IrisXe', 'ResNet18 (R18), ResNet34 (R34)']
Iris_ResNet_net = [
    'R18', 'R18', 'R18', 'R18', 'R18', 'R18', 'R18', 'R18', 'R18', 'R34',
    'R34', 'R34', 'R34'
]
Iris_ResNet_modi = [
    'inf', 'inf', 'train', 'train', 'train', 'train', 'train', 'train',
    'train', 'inf', 'inf', 'inf', 'train'
]
Iris_ResNet_N = [16, 32, 8, 8, 8, 16, 32, 64, 128, 8, 16, 32, 16]
Iris_ResNet_threads = [24, 24, 4, 12, 17, 24, 24, 17, 17, 22, 24, 24, 24]
Iris_ResNet_CPU = [
    55.55, 110.41, 381.0, 178.0, 151.7, 197.7, 394.4, 940.2, 1676.0, 57.7,
    103.1, 193.6, 363.3
]
Iris_ResNet_GPU = [
    58.02, 107.9, 177.8, 177.9, 181.5, 294.1, 572.7, 0, 0, 61.6, 102.3, 189.9,
    449.5
]
Iris_ResNet_MILP = [
    55.24, 107.9, 152.1, 135.5, 136.1, 187.5, 357.9, 857.1, 1608.2, 57.3, 99.7,
    186.0, 341.6
]

# VGG and GoogleNet on platform 'IRIS XE'
Iris_VGG_info = ['IrisXe', 'VGG16 (V16), VGG19 (V19), GoogleNet (G)']
Iris_VGG_net = [
    'V16', 'V16', 'V16', 'V16', 'V16', 'V16', 'V19', 'V19', 'V19', 'G', 'G',
    'G'
]
Iris_VGG_modi = [
    'inf', 'inf', 'inf', 'train', 'train', 'train', 'inf', 'inf', 'train',
    'inf', 'inf', 'inf'
]
Iris_VGG_N = [2, 4, 8, 2, 2, 4, 2, 4, 4, 1, 1, 1]
Iris_VGG_threads = [24, 24, 16, 22, 24, 22, 24, 24, 24, 10, 12, 14]
Iris_VGG_CPU = [
    64.8, 123.6, 287.9, 216.4, 224.9, 387.8, 72.8, 143.2, 478.6, 14.6, 10.7,
    15.1
]
Iris_VGG_GPU = [
    55.4, 94.5, 172.1, 187.9, 187.8, 325.7, 65.5, 114.0, 395.6, 13.6, 13.0,
    13.7
]
Iris_VGG_MILP = [
    54.0, 93.1, 170.9, 168.0, 175.5, 315.0, 64.1, 112.9, 394.5, 11.9, 10.0,
    11.7
]

# ResNet, VGG and GoogleNet on platform 'Gen9'
Gen9_info = ['Gen9', 'ResNet18 (R18), VGG16 (V16), VGG19 (V19)']
Gen9_net = [
    'R18', 'R18', 'R18', 'R18', 'V16', 'V16', 'V16', 'V19', 'V19', 'V19'
]
Gen9_modi = [
    'inf', 'inf', 'inf', 'inf', 'inf', 'inf', 'inf', 'inf', 'inf', 'inf'
]
Gen9_N = [8, 16, 16, 32, 2, 4, 8, 2, 4, 8]
Gen9_threads = [8, 10, 12, 12, 4, 4, 4, 4, 4, 4]
Gen9_CPU = [
    100.9, 167.8, 159.5, 301.2, 175.9, 402.3, 757.6, 253.4, 482.6, 931.9
]
Gen9_GPU = [
    104.9, 184.6, 183.5, 353.9, 233.5, 401.4, 754.7, 277.5, 490.0, 913.8
]
Gen9_MILP = [
    93.8, 152.8, 153.7, 291.0, 164.0, 345.2, 652.9, 218.9, 407.0, 770.6
]


def main():
    # create csv files (draw with tikz in paper)
    toCSV(Iris_ResNet_info, Iris_ResNet_net, Iris_ResNet_modi, Iris_ResNet_N,
          Iris_ResNet_threads, Iris_ResNet_CPU, Iris_ResNet_GPU,
          Iris_ResNet_MILP)
    toCSV(Iris_VGG_info, Iris_VGG_net, Iris_VGG_modi, Iris_VGG_N,
          Iris_VGG_threads, Iris_VGG_CPU, Iris_VGG_GPU, Iris_VGG_MILP)
    toCSV(Gen9_info, Gen9_net, Gen9_modi, Gen9_N, Gen9_threads, Gen9_CPU,
          Gen9_GPU, Gen9_MILP)

    # create plots as .png files
    plot(Iris_ResNet_info, Iris_ResNet_net, Iris_ResNet_modi, Iris_ResNet_N,
         Iris_ResNet_threads, Iris_ResNet_CPU, Iris_ResNet_GPU,
         Iris_ResNet_MILP)
    plot(Iris_VGG_info, Iris_VGG_net, Iris_VGG_modi, Iris_VGG_N,
         Iris_VGG_threads, Iris_VGG_CPU, Iris_VGG_GPU, Iris_VGG_MILP)
    plot(Gen9_info, Gen9_net, Gen9_modi, Gen9_N, Gen9_threads, Gen9_CPU,
         Gen9_GPU, Gen9_MILP)

if __name__ == "__main__":
    main()
