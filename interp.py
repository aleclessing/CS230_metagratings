import scipy.ndimage
import numpy as np


def predict(input_im, scaling=2):
    return scipy.ndimage.zoom(input_im, (1, scaling, scaling), order=1)


if __name__ == '__main__':
    import dataloader_spacetime
    import matplotlib.pyplot as plt

    data = dataloader_spacetime.MetaGratingDataLoader(return_hres=True, n_samp_pts=0)

    hr_im, lr_im = data[3]

    pred_hr_im = predict(lr_im)

    print(pred_hr_im.shape)

    plt.imshow(pred_hr_im[0])
    
    plt.colorbar()

    print(np.max(lr_im))
    print(np.max(pred_hr_im))

    plt.show()
    