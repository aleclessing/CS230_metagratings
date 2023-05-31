import scipy.ndimage
import numpy as np
import dataloader as loader
import matplotlib.pyplot as plt
from predict import predict_img
import jnet
import torch
from utils import plot_hr_lr_sr
from sklearn.metrics import mean_squared_error

def predictWithInterpolation(input_im, scaling=2):
    return scipy.ndimage.zoom(input_im, (1, scaling, scaling), order=1)

def pltCompatisonsSamples(sampleNumber):
    exnum = sampleNumber # example to be plotted
    model = "model1.pth" # SR model to be used
    # Interpolated data
    data = loader.MetaGratingDataLoader(return_hres=True, n_samp_pts=0)
    hr_im, lr_im = data[exnum]
    pred_hr_im = predictWithInterpolation(lr_im)[1:3,:,:] # Slice to keep Real and Imaginary interpolated fields

    # Predicted data by model
    hr_img, lr_img = loader.MetaGratingDataLoader(return_hres=True, n_samp_pts=0)[int(exnum)]
    net = jnet.JNet(im_dim=(64, 256), static_channels=1, dynamic_channels=2)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.to(device=device)
    state_dict = torch.load(model, map_location=device)
    net.load_state_dict(state_dict)
    sr_img = predict_img(net=net, lr_img=lr_img, device=device) # must first open a data file and read it into a numpy array    
    plot_hr_lr_sr(hr_img, lr_img, sr_img,pred_hr_im)


if __name__ == '__main__':

    # pltCompatisonsSamples(1) #### PLOT ONE SAMPLE

    # Evaluate Error from Interpolation and Predictions accross N samples
    N = 100
    x = list(range(N)) #Sample Number Array 
    model = "model1.pth" # SR model to be used
    ya1=[] #mse_pred_real
    ya2=[] #mse_pred_imag
    ya3=[] #mse_interp_real
    ya4=[] #mse_interp_imag
    for i in range(N):
        # Interpolated data
        data = loader.MetaGratingDataLoader(return_hres=True, n_samp_pts=0)
        hr_im, lr_im = data[i]
        pred_hr_im = predictWithInterpolation(lr_im)[1:3,:,:] # Slice to keep Real and Imaginary interpolated fields
        # Predicted data by model
        hr_img, lr_img = loader.MetaGratingDataLoader(return_hres=True, n_samp_pts=0)[int(i)]
        net = jnet.JNet(im_dim=(64, 256), static_channels=1, dynamic_channels=2)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        net.to(device=device)
        state_dict = torch.load(model, map_location=device)
        net.load_state_dict(state_dict)
        sr_img = predict_img(net=net, lr_img=lr_img, device=device) # must first open a data file and read it into a numpy array
        ### CALCULATE ERROR ACCROSS A NUMBER OF SAMPLES
        mse_pred_real = mean_squared_error(sr_img[0], hr_img[0]) #Real Fields Predicted
        mse_pred_imag = mean_squared_error(sr_img[1], hr_img[1]) #Imag Fields Predicted
        mse_interp_real = mean_squared_error(pred_hr_im[0], hr_img[0]) #Real Fields Interpolated
        mse_interp_imag = mean_squared_error(pred_hr_im[1], hr_img[1]) #Imag Fields Interpolated
        ya1.append(mse_pred_real)
        ya2.append(mse_pred_imag)
        ya3.append(mse_interp_real)
        ya4.append(mse_interp_imag)

    #plot 1: Real Fields
    xarr = np.array([[x],[x]])
    yarr = np.array([[ya1],[ya3]])
    for i in range(2):
        plt.subplot(1, 2, 1)
        plt.plot(xarr[i,0], yarr[i,0])
    plt.xlabel("Sample Number")
    plt.ylabel("MSE Error")
    plt.title("Real Fields")
    plt.gca().legend(('mse_pred_real','mse_interp_real'))

    #plot 2: Imaginary Fields
    xarr = np.array([[x],[x]])
    yarr = np.array([[ya2],[ya4]])
    for i in range(2):
        plt.subplot(1, 2, 2)
        plt.plot(xarr[i,0], yarr[i,0])
    plt.xlabel("Sample Number")
    plt.ylabel("MSE Error")
    plt.title("Imaginary Fields")
    plt.gca().legend(('mse_pred_imag','mse_interp_imag'))
    
    plt.suptitle("MSE Errors for Predicted and Interpolated Values")
    plt.show()
