import matplotlib.pyplot as plt

def plot_hr_lr_sr(hr_data, lr_data, sr_data, int_data=None):
    """
    Plots the HR ground truth, LR image, and HR super-resolved (SR) image 
    side-by-side for both the real and imaginary parts of the Hy field.
    Want two columns (real and imaginary) and three rows (HR, LR, and SR).

    Args:
        lr_data: the LR array for single example; shape [3, 32, 128] np array
        hr_data: the HR array for single example; shape [2, 64, 256] np array
        Note: the 3 channels are permitivities and the Re and Im parts of the 
        Hy field.
    """
    if not int_data:
        num_rows = 3
    else:
        num_rows = 4

    fig, axs = plt.subplots(num_rows, 2, figsize=(10, 10))
    fig.suptitle('HR, LR, and SR Hy field')
    axs[0, 0].imshow(hr_data[0, :, :])
    axs[0, 0].set_title('HR Re(Hy)')
    axs[0, 1].imshow(hr_data[1, :, :])
    axs[0, 1].set_title('HR Im(Hy)')
    axs[1, 0].imshow(lr_data[1, :, :])
    axs[1, 0].set_title('LR Re(Hy)')
    axs[1, 1].imshow(lr_data[2, :, :])
    axs[1, 1].set_title('LR Im(Hy)')
    axs[2, 0].imshow(sr_data[0, :, :])
    axs[2, 0].set_title('SR Re(Hy)')
    axs[2, 1].imshow(sr_data[1, :, :])
    axs[2, 1].set_title('SR Im(Hy)')
    if int_data:
        axs[3, 0].imshow(int_data[0, :, :])
        axs[3, 0].set_title('Int Re(Hy)')
        axs[3, 1].imshow(int_data[1, :, :])
        axs[3, 1].set_title('Int Im(Hy)')

    plt.savefig('pred.png')
    plt.show()
