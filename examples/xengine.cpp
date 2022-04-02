#include "../src/core/network.cpp"
#include <filesystem>

using namespace std;

void createOrRunSchedule(Network &net, const string &images,
                         const string &labels) {
  const string schedulefile =
      "../data/schedules/" + net.name() + "_" + net.mode() + "_schedule.txt";
  if (!checkIfFileExists(schedulefile)) {
    net.createSchedule(schedulefile, images, labels);
  } else {
    net.runSchedule(schedulefile, images, labels, 1);
  }
}

void runILPSolver(Network &net, const string &images, const string &labels) {
  const string output_filename =
      net.output_directory() + "/" + net.name() + "_" + net.mode();
  const string mpsfile = output_filename + ".mps";
  const string logfile = output_filename + ".log";
  const int benchmark = 1;
  net.solveILP(mpsfile, logfile, images, labels, benchmark);
}

void execute_network(const string &model_name, string &images, string &labels,
                     const string &device_file, const int &training,
                     const string &output_dir) {
  const string model = "../data/models/" + model_name + ".onnx";
  const int verbose = 1;
  Network net =
      Network(model_name, model, device_file, training, output_dir, verbose);
  if (!filesystem::exists(images) || !filesystem::exists(labels)) {
    images = "";
    labels = "";
    cout << "WARNING: no real image and label data available - using dummy "
            "data instead!"
         << endl;
  }
  createOrRunSchedule(net, images, labels);
  runILPSolver(net, images, labels);
}

void lenet(const int &batchsize, const int &training, const string &device_file,
           const string &output_dir) {
  const string model_name = "lenet_bs" + to_string(batchsize);
  string images = training
                      ? "../data/datasets/mnist_train/train-images-idx3-ubyte"
                      : "../data/datasets/mnist_test/t10k-images-idx3-ubyte";
  string labels = training
                      ? "../data/datasets/mnist_train/train-labels-idx1-ubyte"
                      : "../data/datasets/mnist_test/t10k-labels-idx1-ubyte";
  execute_network(model_name, images, labels, device_file, training,
                  output_dir);
}

void vgg(const string &version, const int &batchsize, const int &training,
         const string &device_file, const string &output_dir) {
  const string model_name = "vgg" + version + "-7_bs" + to_string(batchsize);
  string images = "../data/datasets/imagenet_test/images-idx4-ubyte";
  string labels = "../data/datasets/imagenet_test/labels-idx1-short";
  execute_network(model_name, images, labels, device_file, training,
                  output_dir);
}

void resnet(const string &version, const int &batchsize, const int &training,
            const string &device_file, const string &output_dir) {
  const string model_name =
      "resnet" + version + "-v1-7_bs" + to_string(batchsize);
  string images = "../data/datasets/imagenet_test/images-idx4-ubyte";
  string labels = "../data/datasets/imagenet_test/labels-idx1-short";
  execute_network(model_name, images, labels, device_file, training,
                  output_dir);
}

void googlenet(const int &batchsize, const int &training,
               const string &device_file, const string &output_dir) {
  const string model_name = "googlenet-7_bs" + to_string(batchsize);
  string images = "../data/datasets/imagenet_test/images-idx4-ubyte";
  string labels = "../data/datasets/imagenet_test/labels-idx1-short";
  execute_network(model_name, images, labels, device_file, training,
                  output_dir);
}

void unet(const string &version, const int &batchsize, const int &training,
          const string &device_file, const string &output_dir) {
  const string model_name = "unet" + version + "_bs" + to_string(batchsize);
  string images = "";
  string labels = "";
  execute_network(model_name, images, labels, device_file, training,
                  output_dir);
}

int run(const string &name, const int &batchsize, const int &training,
        const string &device_file, const string &output_dir) {
  if (name.compare("lenet") == 0) {
    lenet(batchsize, training, device_file, output_dir);
  } else if (name.compare("vgg16") == 0) {
    vgg("16", batchsize, training, device_file, output_dir);
  } else if (name.compare("vgg19") == 0) {
    vgg("19", batchsize, training, device_file, output_dir);
  } else if (name.compare("resnet18") == 0) {
    resnet("18", batchsize, training, device_file, output_dir);
  } else if (name.compare("resnet50") == 0) {
    resnet("50", batchsize, training, device_file, output_dir);
  } else if (name.compare("resnet34") == 0) {
    resnet("34", batchsize, training, device_file, output_dir);
  } else if (name.compare("googlenet") == 0) {
    googlenet(batchsize, training, device_file, output_dir);
  } else if (name.compare("unet2D") == 0) {
    unet("2D", batchsize, training, device_file, output_dir);
  } else if (name.compare("unet3D") == 0) {
    unet("3D", batchsize, training, device_file, output_dir);
  } else {
    throw runtime_error("Unknown NETWORK option!");
  }
  return 0;
}

int main(const int argc, const char **argv) {
  if (argc < 5) {
    throw runtime_error(
        "usage: ./examples/xengine NETWORK "
        "(lenet|vgg16|vgg19|resnet18|resnet34|resnet50|unet|googlenet) "
        "BATCHSIZE "
        "MODE (0=test|1=train) "
        "OUTPUT_DIRECTORY (use '.' for default directory) "
        "DEVICES (optional device file, default: "
        "'../data/devices/devices_auto.txt')");
  }
  const string name = argv[1];
  const int batchsize = stoi(argv[2]);
  if (batchsize <= 0) {
    throw runtime_error("BATCHSIZE must be > 0!");
  }
  const int training = stoi(argv[3]);
  if (training < 0 || training > 1) {
    throw runtime_error("MODE must be either 0 or 1! (0=test, 1=train)");
  }
  const string output_dir = argv[4];
  if (output_dir.find(".txt") != string::npos) {
    throw runtime_error("OUTPUT DIRECTORY seems to have been skipped! Did you "
                        "provide a device file here?");
  }
  if (!filesystem::exists(output_dir)) {
    filesystem::create_directory(output_dir);
  }
  if (!filesystem::is_directory(output_dir)) {
    throw runtime_error(output_dir + " is NO directory!");
  }
  const string device_file =
      argc == 6 ? argv[5] : "../data/devices/devices_auto.txt";
  if (device_file.find(".txt") == string::npos) {
    throw runtime_error("Device file must end with '.txt'");
  }

  run(name, batchsize, training, device_file, output_dir);

  return 0;
}
