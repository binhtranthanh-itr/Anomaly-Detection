import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    with tf.device('/cpu:0'):
        for i in range(0, 15):
            img_1d = np.loadtxt("/home/ttb/Documents/CNN/CNN_New/result/data-" + str(i) + ".txt")
            img = np.reshape(img_1d, (48, 40))
            recon_img_1d = np.loadtxt("/home/ttb/Documents/CNN/CNN_New/result/Out-" + str(i) + ".txt")
            recon_img = np.reshape(recon_img_1d, (48, 40))
            
            #Draw with arr 1D
            # plt.plot(img_1d, label='origin')
            # plt.plot(recon_img_1d, label='recon_img')
            
            # Draw with arr 2D
            plt.subplot(1, 2, 1)
            plt.imshow(img)
            plt.title('origin')
            plt.subplot(1, 2, 2)
            plt.imshow(recon_img)
            plt.title('reconstruct')
            plt.legend()
            plt.show()
