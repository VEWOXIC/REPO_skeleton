import torch
import matplotlib.pyplot as plt
import torch.fft as fft
import math
def lowpass_torch(input, limit):
    pass1 = torch.abs(fft.rfftfreq(960)) < limit
    print(pass1)
    fft_input = fft.rfft(input)
    # plt.plot(range(len(fft_input)),fft_input)
    return fft.irfft(fft_input * pass1)

import statsmodels.api as sm
x = 2*torch.cos(torch.arange(96)/(2*math.pi)) + 0.5*torch.cos(torch.arange(96)/(4*math.pi)) + torch.arange(96)*0.1
rd = sm.tsa.seasonal_decompose(x,model='additive', extrapolate_trend='freq',period=21)
lx = rd.trend
# x = 2*torch.cos(torch.arange(960)/(20*math.pi)) + 0.5*torch.cos(torch.arange(960)/(200*math.pi)) + torch.arange(960)*0.1
# lx = lowpass_torch(x, 1/19)
plt.plot(range(len(x)),x,label='x')
plt.plot(range(len(lx)),lx,label='lx')
plt.plot(range(len(x-lx)),x-lx,label='hx')
plt.legend()
plt.show()
# print(x.shape)