import numpy as np

# Helpers: im2col / col2im
def im2col(x, filter_h, filter_w, stride=1, pad=0):
    N, C, H, W = x.shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1

    xp = np.pad(x, ((0,0),(0,0),(pad,pad),(pad,pad)), mode='constant')
    cols = np.zeros((N, C, filter_h, filter_w, out_h, out_w), dtype=x.dtype)

    for y in range(filter_h):
        y_max = y + stride*out_h
        for x0 in range(filter_w):
            x_max = x0 + stride*out_w
            cols[:, :, y, x0, :, :] = xp[:, :, y:y_max:stride, x0:x_max:stride]

    cols = cols.transpose(0,4,5,1,2,3).reshape(N*out_h*out_w, -1)
    return cols

def col2im(cols, x_shape, filter_h, filter_w, stride=1, pad=0):
    N, C, H, W = x_shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1