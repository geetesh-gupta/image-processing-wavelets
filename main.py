import cv2 
import numpy as np
from math import sqrt

def processImage(image):
    image = cv2.imread(image) 
    image = cv2.resize(image,(512,512))
    image = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2GRAY) 
    return image

def convolve2D(image, kernel, padding=0, strides=1):
    kernel = np.flipud(np.fliplr(kernel))   # Cross Correlation

    # Gather Shapes of Kernel + Image + Padding
    kernelCols = kernel.shape[0]
    kernelRows = kernel.shape[1]
    imageCols = image.shape[0]
    imageRows = image.shape[1]

    # Shape of Output Convolution
    xOutput = int(((imageCols - kernelCols + 2 * padding) / strides) + 1)
    yOutput = int(((imageRows - kernelRows + 2 * padding) / strides) + 1)
    output = np.zeros((xOutput, yOutput))

    # Apply Equal Padding to All Sides
    if padding != 0:
        imagePadded = np.zeros((image.shape[0] + padding*2, image.shape[1] + padding*2))
        imagePadded[int(padding):int(-1 * padding), int(padding):int(-1 * padding)] = image
    else:
        imagePadded = image

    for y in range(image.shape[1]):
        if y > image.shape[1] - kernelRows:
            break
        if y % strides == 0:
            for x in range(image.shape[0]):
                if x > image.shape[0] - kernelCols:
                    break
                if x % strides == 0:
                    output[x, y] = (kernel * imagePadded[x: x + kernelCols, y: y + kernelRows]).sum()
    return output

def downsampleImage(image, rowScale=1, colScale=1):
    numRows = image.shape[1]
    numCols = image.shape[0]
    outputNumRows = int(numRows / rowScale)
    outputNumCols = int(numCols / colScale)
    output = np.zeros((outputNumCols, outputNumRows))
    for i in range(outputNumRows):
        for j in range(outputNumCols):
            output[j][i] = image[j*colScale][i*rowScale]
    return output

def upsampleImage(image, rowScale=1, colScale=1):
    numRows = image.shape[1]
    numCols = image.shape[0]
    outputNumRows = int(numRows * rowScale)
    outputNumCols = int(numCols * colScale)
    output = np.zeros((outputNumCols, outputNumRows))
    for i in range(numRows):
        for j in range(numCols):
            output[j*colScale][i*rowScale] = image[j][i]
    return output

def convoluteAndDownsample(image, filterr, rowScale=1, colScale=1):
    convolutedImage = convolve2D(image, filterr)
    downsampledImage = downsampleImage(convolutedImage, rowScale, colScale)
    return downsampledImage

def upsampleAndConvolute(image, filterr, rowScale=1, colScale=1):
    upsampledImage = upsampleImage(image, rowScale, colScale)
    convolutedImage = convolve2D(upsampledImage, filterr)
    return convolutedImage

def decomposeImage(image, lpf, hpf):
    # Rowwise Convolution and Downsampling
    L = convoluteAndDownsample(image, lpf, 1, 2)
    H = convoluteAndDownsample(image, hpf, 1, 2)

    # Columnwise Convolution and Downsampling for each
    LL = convoluteAndDownsample(L, lpf.T, 2)
    LH = convoluteAndDownsample(L, hpf.T, 2)
    HL = convoluteAndDownsample(H, lpf.T, 2)
    HH = convoluteAndDownsample(H, hpf.T, 2)

    return [LL, LH, HL, HH]

def reconstructImage(image, lpf, hpf, coeffs):
    LL, LH, HL, HH = coeffs

    # Columnwise Downsampling and Convolution for each
    LLr = upsampleAndConvolute(LL, lpf.T, 1, 2)
    LHr = upsampleAndConvolute(LH, hpf.T, 1, 2)
    HLr = upsampleAndConvolute(HL, lpf.T, 1, 2)
    HHr = upsampleAndConvolute(HH, hpf.T, 1, 2)
    
    # Rowwise Downsampling and Convolution
    Lr = np.add(LLr, LHr)
    Hr = np.add(HLr, HHr)

    Lrr = upsampleAndConvolute(Lr, lpf, 2)
    Hrr = upsampleAndConvolute(Hr, hpf, 2)
    
    reconstructedImage = np.add(Lrr, Hrr)
    return reconstructedImage

def dwt1(image):
    filterVal = 1/sqrt(2)
    lpf = np.array([[filterVal, filterVal]])
    hpf = np.array([[-filterVal, filterVal]])

    return decomposeImage(image, lpf, hpf)

def dwt2(image):
    lpf = np.array([[-0.12940952255092145, 0.22414386804185735, 0.836516303737469, 0.48296291314469025]])
    hpf = np.array([[-0.48296291314469025, 0.836516303737469, -0.22414386804185735, -0.12940952255092145]])

    return decomposeImage(image, lpf, hpf)

def idwt2(image, coeffs):
    lpf = np.array([[0.48296291314469025, 0.836516303737469, 0.22414386804185735, -0.12940952255092145]])
    hpf = np.array([[-0.12940952255092145, -0.22414386804185735, 0.836516303737469, -0.48296291314469025]])

    return reconstructImage(image, lpf, hpf, coeffs)

def getResultsFromInbuiltLibrary(image):
    import pywt
    coeffs = pywt.dwt2(image, 'db2')
    cA, (cH, cV, cD) = coeffs

    cv2.imwrite("lib_output/LL.jpg",cA)
    cv2.imwrite("lib_output/LH.jpg",cH)
    cv2.imwrite("lib_output/HL.jpg",cV)
    cv2.imwrite("lib_output/HH.jpg",cD)

    return coeffs

def getResultsFromCustomFunction(image):
    coeffs = dwt2(image)
    LL, LH, HL, HH = coeffs
    reconstructedImage = idwt2(image, coeffs)

    cv2.imwrite('custom_output/LL.jpg', LL)
    cv2.imwrite('custom_output/LH.jpg', LH)
    cv2.imwrite('custom_output/HL.jpg', HL)
    cv2.imwrite('custom_output/HH.jpg', HH)
    cv2.imwrite('custom_output/reconstructedImage.jpg', reconstructedImage)

image = processImage('sheldon.png')
getResultsFromInbuiltLibrary(image)
getResultsFromCustomFunction(image)