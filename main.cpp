#include "headers/simple_nn.h"
#include "headers/config.h"
#include <thread>
using namespace std;
using namespace simple_nn;
using namespace Eigen;

int main(int argc, char** argv)
{
	Config cfg;
	cfg.parse(argc, argv);

	int n_train = 7820, n_test = 150, ch = 8, h = 24, w = 20;

	char nameData[25];
	char nameOut[25];

	MatXf latent, test_latent;
	MatXf spectral, test_spectral;
	MatXf test, YM;
	MatXf resultModel;
	DataLoader train_loader, test_loader;

	latent = read_latent(cfg.data_dir, "latentV2.npy", n_train);
	spectral = read_spectral(cfg.data_dir, "spectralV2.npy", n_train);
	train_loader.load(latent, spectral, cfg.batch, ch, h, w, cfg.shuffle_train);

	test_latent = read_latent(cfg.data_dir, "latentV2.npy", n_test);
	test_spectral = read_spectral(cfg.data_dir, "spectralV2.npy", n_test);
	test_loader.load(test_latent, test_spectral, cfg.batch, ch, h, w, cfg.shuffle_test);

	cout << "Dataset loaded." << endl;

	SimpleNN model;

	model.add(new Conv2d(8, 4, 3, 1, cfg.init));
	model.add(new ReLU);
	model.add(new Flatten);

	cout << "Model construction completed." << endl;

	if (cfg.mode == "train") {
		model.compile({ cfg.batch, ch, h, w }, new Adam(cfg.lr, cfg.decay), new MAELoss);
		model.fit(train_loader, cfg.epoch, test_loader);
		model.save("./model_new", cfg.model + ".pth");
	}
	else{
		model.compile({ cfg.batch, ch, h, w });
		model.load(cfg.save_dir, cfg.pretrained);
		for (int i = 0; i < 20; i++){
			 test = test_loader.get_x(i);
			 YM = test_loader.get_y_matrix(i);
			 resultModel = model.evaluate(test);
			 sprintf(nameData, "./result/data-%d.txt", i);
			 sprintf(nameOut, "./result/Out-%d.txt", i);
			 ofstream fileData (nameData);
			 ofstream fileOut (nameOut);
			 if (fileData.is_open() && fileOut.is_open()){
				 fileData << YM;
				 fileOut << resultModel;
				 fileData.close();
				 fileOut.close();
			 }
		 }
	 }

	return 0;
}

