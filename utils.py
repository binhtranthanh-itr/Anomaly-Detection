import numpy as np
from warnings import warn


def _im_to_rows(x, filter_shape, dilation, stride, dilated_shape, res_shape):
    """
    Converts the 4D image to a form such that convolution can be performed via matrix multiplication
    :param x: The image of the dimensions (batch, channels, height, width)
    :param filter_shape: The shape of the filter (num_filters, depth, height, width)
    :param dilation: The dilation for the filter
    :param stride: The stride for the filter
    :param dilated_shape: The dilated shape of the filter
    :param res_shape: The shape of the expected result
    :return: The transformed image
    """
    dilated_rows, dilated_cols = dilated_shape
    num_rows, num_cols = res_shape
    res = np.zeros((x.shape[0], num_rows * num_cols, filter_shape[1], filter_shape[2], filter_shape[3]), dtype=x.dtype)
    for i in range(num_rows):
        for j in range(num_cols):
            res[:, i * num_cols + j, :, :, :] = x[:, :, i * stride[0]:i * stride[0] + dilated_rows:dilation,
                                                j * stride[1]:j * stride[1] + dilated_cols:dilation]
    return res.reshape((res.shape[0], res.shape[1], -1))


def _backward_im_to_rows(top_grad, inp_shape, filter_shape, dilation, stride, dilated_shape, res_shape):
    """
    Gradient transformation for the im2rows operation
    :param top_grad: The grad from the next layer
    :param inp_shape: The shape of the input image
    :param filter_shape: The shape of the filter (num_filters, depth, height, width)
    :param dilation: The dilation for the filter
    :param stride: The stride for the filter
    :param dilated_shape: The dilated shape of the filter
    :param res_shape: The shape of the expected result
    :return: The reformed gradient of the shape of the image
    """
    dilated_rows, dilated_cols = dilated_shape
    num_rows, num_cols = res_shape
    res = np.zeros(inp_shape, dtype=top_grad.dtype)
    top_grad = top_grad.reshape(
        (top_grad.shape[0], top_grad.shape[1], filter_shape[1], filter_shape[2], filter_shape[3]))
    for it in range(num_rows * num_cols):
        i = it // num_rows
        j = it % num_rows
        res[:, :, i * stride[0]:i * stride[0] + dilated_rows:dilation,
            j * stride[1]:j * stride[1] + dilated_cols:dilation] += top_grad[:, it, :, :, :]
    return res


try:
    from numba import jit

    _im_to_rows = jit(_im_to_rows)
    _backward_im_to_rows = jit(_backward_im_to_rows)
except ModuleNotFoundError:
    warn("Numba not found, convolutions will be slow.")


def _filter_to_mat(f):
    """
    Converts a filter to matrix form
    :param f: The filter (num_filters, depth, height, width)
    :return: The matrix form of the filter which can be multiplied
    """
    return f.reshape(f.shape[0], -1).T


def _convolved_to_im(im, res_shape):
    """
    Reshapes the convolved matrix to the shape of the image
    :param im: The convolved matrix
    :param res_shape: The expected shape of the result
    :return: The reshaped image
    """
    im = im.transpose((0, 2, 1))
    return im.reshape(im.shape[0], im.shape[1], res_shape[0], res_shape[1])


def conv2d(image, filters, dilation, stride):
    """
    Performs a 2D convolution on the image given the filters
    :param image: The input image (batch, channel, height, width)
    :param filters: The filters (num_filters, depth, height, width)
    :param dilation: The dilation factor for the filter
    :param stride: The stride for convolution
    :return: The convolved image
    """
    filter_shape = filters.shape
    im_shape = image.shape
    dilated_shape = ((filter_shape[2] - 1) * dilation + 1, (filter_shape[3] - 1) * dilation + 1)
    res_shape = ((im_shape[2] - dilated_shape[0]) // stride[0] + 1, (im_shape[3] - dilated_shape[1]) // stride[1] + 1)
    imrow = _im_to_rows(image, filters.shape, dilation, stride, dilated_shape, res_shape)
    filtmat = _filter_to_mat(filters)
    res = imrow.dot(filtmat)
    return _convolved_to_im(res, res_shape)


def backward_conv2d(top_grad, image, filters, dilation, stride):
    """
    Given the grads from the next op, performs the backward convolution pass
    :param top_grad: The grad from the next op
    :param image: The input image to this operation
    :param filters: The filters for this operation
    :param dilation: The dilation factor for the filter
    :param stride: The stride for the convolution
    :return: A tuple representing the grads wrt the input image and the filters
    """
    filter_shape = filters.shape
    im_shape = image.shape
    dilated_shape = ((filter_shape[2] - 1) * dilation + 1, (filter_shape[3] - 1) * dilation + 1)
    res_shape = ((im_shape[2] - dilated_shape[0]) // stride[0] + 1, (im_shape[3] - dilated_shape[1]) // stride[1] + 1)
    imrow = _im_to_rows(image, filters.shape, dilation, stride, dilated_shape, res_shape)
    filtmat = _filter_to_mat(filters)
    gradmat = top_grad.reshape((top_grad.shape[0], top_grad.shape[1], -1)).transpose((0, 2, 1))
    filt_grad = np.matmul(imrow.transpose((0, 2, 1)), gradmat).sum(axis=0).T.reshape(filter_shape)
    inp_grad_mat = gradmat.dot(filtmat.T)
    inp_grad = _backward_im_to_rows(inp_grad_mat, image.shape, filters.shape, dilation,
                                    stride, dilated_shape, res_shape)
    return inp_grad, filt_grad


def reshape(x, new_shape):
    """
    Reshape the input to the new shape (preserves the batch)
    :param x: The input
    :param new_shape: The new shape
    :return: The reshaped tensor
    """
    old_shape = x.shape[1:]
    return x.reshape((x.shape[0], *new_shape)), old_shape


def backward_reshape(top_grad, old_shape):
    """
    Perform the backward pass on the reshape operation
    :param top_grad: The gradient from the next layer
    :param old_shape: The old shape
    :return: The gradient for the input
    """
    return top_grad.reshape((top_grad.shape[0], *old_shape))


def mse(x, y):
    """
    Sum of squared error between two tensors. Average across the batch.
    :param x: The input tensor
    :param y: The target tensor
    :return: The squared error
    """
    return ((x - y) ** 2).mean()


def backward_mse(top_grad, x, y):
    """
    Get the gradient with respect to x.
    :param top_grad: The gradient from the next layer
    :param x: The input
    :param y: The target
    :return: The grad wrt x
    """
    return top_grad * (2 * (x - y)) / x.shape[0]
