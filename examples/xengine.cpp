#include "../src/core/network.cpp"
#include <filesystem>

using namespace std;

void execute_network(const string &model_name, const string &images,
                     const string &labels, const string &schedule_file,
                     const string &devices, const int training,
                     const string output_dir) {
  auto model = "../data/models/" + model_name + ".onnx";
  const int verbose = 1;
  const size_t num_iterations = 1;
  Network net = Network(model_name, model, devices, training, verbose);
  net.init();
  if (verbose > 0) {
    cout << "Simple optimizer ..." << endl;
  }
  if (!checkIfFileExists(schedule_file)) {
    net.benchmark(images, labels);
    net.writeScheduleFile(schedule_file);
  }
  net.setSchedule(schedule_file);
  net.run(images, labels, num_iterations);
  if (verbose > 0) {
    cout << "ILP optimizer ..." << endl;
  }
  const string mode = training ? "training" : "inference";
  if (!filesystem::exists(output_dir)) {
    filesystem::create_directory(output_dir);
  }
  if (!filesystem::is_directory(output_dir)) {
    throw runtime_error(output_dir + " is NO directory!");
  }
  const string output_filename = output_dir + "/" + model_name + "_" + mode;
  const string mpsfile = output_filename + ".mps";
  const string logfile = output_filename + ".log";
  const int benchmarkILP = 1;
  net.solveILP(mpsfile, logfile, images, labels, benchmarkILP);
}

void lenet(const int batchsize, const int training, const string devices,
           const string output_dir) {
  string model_name, images, labels, schedule_file;
  model_name = "lenet_bs" + to_string(batchsize);
  if (training) {
    images = "../data/datasets/mnist_train/train-images-idx3-ubyte";
    labels = "../data/datasets/mnist_train/train-labels-idx1-ubyte";
    schedule_file = "../data/schedules/lenet_train_schedule.txt";
  } else {
    images = "../data/datasets/mnist_test/t10k-images-idx3-ubyte";
    labels = "../data/datasets/mnist_test/t10k-labels-idx1-ubyte";
    schedule_file = "../data/schedules/lenet_inf_schedule.txt";
  }
  execute_network(model_name, images, labels, schedule_file, devices, training,
                  output_dir);
}

void vgg16(const int batchsize, const int training, const string devices,
           const string output_dir) {
  string model_name, images, labels, schedule_file;
  model_name = "vgg16-7_bs" + to_string(batchsize);
  images = "../data/datasets/imagenet_test/images-idx4-ubyte";
  labels = "../data/datasets/imagenet_test/labels-idx1-short";
  if (training) {
    schedule_file = "../data/schedules/vgg16-7_train_schedule.txt";
  } else {
    schedule_file = "../data/schedules/vgg16-7_inf_schedule.txt";
  }
  execute_network(model_name, images, labels, schedule_file, devices, training,
                  output_dir);
}

void resnet(const string version, const int batchsize, const int training,
            const string devices, const string output_dir) {
  string model_name, images, labels, schedule_file;
  model_name = "resnet" + version + "-v1-7_bs" + to_string(batchsize);
  images = "../data/datasets/imagenet_test/images-idx4-ubyte";
  labels = "../data/datasets/imagenet_test/labels-idx1-short";
  if (training) {
    schedule_file = "../data/schedules/resnet50-v1-7_train_schedule.txt";
  } else {
    schedule_file = "../data/schedules/resnet50-v1-7_inf_schedule.txt";
  }
  execute_network(model_name, images, labels, schedule_file, devices, training,
                  output_dir);
}

void unet(const int batchsize, const int training, const string devices,
          const string output_dir) {
  string model_name, images, labels, schedule_file;
  model_name = "unet";
  images = "";
  labels = "";
  if (training) {
    schedule_file = "../data/schedules/unet_train_schedule.txt";
  } else {
    schedule_file = "../data/schedules/unet_inf_schedule.txt";
  }
  execute_network(model_name, images, labels, schedule_file, devices, training,
                  output_dir);
}

int run(const string name, const int batchsize, const int training,
        const string devices, const string output_dir) {
  if (name.compare("lenet") == 0) {
    lenet(batchsize, training, devices, output_dir);
  } else if (name.compare("vgg16") == 0) {
    vgg16(batchsize, training, devices, output_dir);
  } else if (name.compare("resnet18") == 0) {
    resnet("18", batchsize, training, devices, output_dir);
  } else if (name.compare("resnet50") == 0) {
    resnet("50", batchsize, training, devices, output_dir);
  } else if (name.compare("unet") == 0) {
    unet(batchsize, training, devices, output_dir);
  } else {
    throw runtime_error("Unknown NETWORK option!");
    return 1;
  }
  return 0;
}

int main(const int argc, const char **argv) {
  if (argc < 5) {
    cout << "usage: ./examples/xengine NETWORK "
            "(lenet|vgg16|resnet18|resnet50) "
            "BATCHSIZE "
            "MODE (0=test|1=train) "
            "OUTPUT_DIRECTORY (use '.' for default directory) "
            "DEVICES (optional device file, default: "
            "'../data/devices/devices_auto.txt')"
         << endl;
    return 1;
  }
  int batchsize = stoi(argv[2]);
  if (batchsize <= 0) {
    throw runtime_error("BATCHSIZE must be > 0!");
  }
  int training = stoi(argv[3]);
  if (training < 0 || training > 1) {
    throw runtime_error("MODE must be either 0 or 1! (0=test, 1=train)");
  }
  const string output_dir = argv[4];
  if (output_dir.find(".txt") != string::npos) {
    throw runtime_error("OUTPUT DIRECTORY seems to have been skipped! Did you "
                        "provide a device file here?");
  }
  const string devices =
      argc == 6 ? argv[5] : "../data/devices/devices_auto.txt";
  if (devices.find(".txt") == string::npos) {
    throw runtime_error("Device file must end with '.txt'");
  }
  run(argv[1], batchsize, training, devices, output_dir);
  return 1;
}
