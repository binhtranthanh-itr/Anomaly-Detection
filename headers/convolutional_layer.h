#pragma once
#include "layer.h"

namespace simple_nn
{
	class Conv2d : public Layer
	{
	private:
		int batch;
		int ic;
		int oc;
		int ih;
		int iw;
		int ihw;
		int oh;
		int ow;
		int ohw;
		int kh;
		int kw;
		int pad;
		string option;
		MatXf dkernel, mkernel, vkernel, m_hat_kernel, v_hat_kernel;
		VecXf dbias, mbias, vbias, m_hat_bias, v_hat_bias;
		MatXf im_col;
	public:
		MatXf kernel;
		VecXf bias;
		Conv2d(int in_channels, int out_channels, int kernel_size, int padding,
			string option);
		void set_layer(const vector<int>& input_shape) override;
		void forward(const MatXf& prev_out, bool is_training) override;
		void backward(const MatXf& prev_out, MatXf& prev_delta) override;
		void update_weight(float lr, float decay) override;
		void zero_grad() override;
		vector<int> output_shape() override;
	};

	Conv2d::Conv2d(
		int in_channels,
		int out_channels,
		int kernel_size,
		int padding,
		string option
	) :
		Layer(LayerType::CONV2D),
		batch(0),
		ic(in_channels),
		oc(out_channels),
		ih(0),
		iw(0),
		ihw(0),
		oh(0),
		ow(0),
		ohw(0),
		kh(kernel_size),
		kw(kernel_size),
		pad(padding),
		option(option) {}

	void Conv2d::set_layer(const vector<int>& input_shape)
	{
		batch = input_shape[0];
		ic = input_shape[1];
		ih = input_shape[2];
		iw = input_shape[3];
		ihw = ih * iw;
		oh = calc_outsize(ih, kh, 1, pad);
		ow = calc_outsize(iw, kw, 1, pad);
		ohw = oh * ow;

		output.resize(batch * oc, ohw);
		delta.resize(batch * oc, ohw);
		kernel.resize(oc, ic * kh * kw);
		dkernel.resize(oc, ic * kh * kw);
		mkernel.resize(oc, ic * kh * kw);
		vkernel.resize(oc, ic * kh * kw);
		m_hat_kernel.resize(oc, ic * kh * kw);
		v_hat_kernel.resize(oc, ic * kh * kw);
		bias.resize(oc);
		dbias.resize(oc);
		mbias.resize(oc);
		vbias.resize(oc);
		m_hat_bias.resize(oc);
		v_hat_bias.resize(oc);
		im_col.resize(ic * kh * kw, ohw);

		int fan_in = kh * kw * ic;
		int fan_out = kh * kw * oc;
		init_weight(kernel, fan_in, fan_out, option);
		bias.setZero();
	}

	void Conv2d::forward(const MatXf& prev_out, bool is_training)
	{
		for (int n = 0; n < batch; n++) {
			const float* im = prev_out.data() + (ic * ihw) * n;
			im2col(im, ic, ih, iw, kh, 1, pad, im_col.data());
			output.block(oc * n, 0, oc, ohw).noalias() = kernel * im_col;
			output.block(oc * n, 0, oc, ohw).colwise() += bias;
		}
	}

	void Conv2d::backward(const MatXf& prev_out, MatXf& prev_delta)
	{
		for (int n = 0; n < batch; n++) {
			const float* im = prev_out.data() + (ic * ihw) * n;
			im2col(im, ic, ih, iw, kh, 1, pad, im_col.data());
			dkernel += delta.block(oc * n, 0, oc, ohw) * im_col.transpose();
			dbias += delta.block(oc * n, 0, oc, ohw).rowwise().sum();
		}
		if (!is_first) {
			for (int n = 0; n < batch; n++) {
				float* begin = prev_delta.data() + ic * ihw * n;
				im_col = kernel.transpose() * delta.block(oc * n, 0, oc, ohw);
				col2im(im_col.data(), ic, ih, iw, kh, 1, pad, begin);
			}
		}
	}

	void Conv2d::update_weight(float lr, float decay)
	{
		float beta1 = 0.9f;
		float beta2 = 0.999f;
		float epsilon = 1e-8;

		mkernel = beta1 * mkernel.array() + (1 - beta1) * (dkernel/batch).array();
		vkernel = beta2 * vkernel.array() + (1 - beta2) * (dkernel/batch).array().square();
		mbias = beta1 * mbias.array() + (1 - beta1) * (dbias/batch).array();
		vbias = beta2 * vbias.array() + (1 - beta2) * (dbias/batch).array().square();

		m_hat_kernel = mkernel / (1 - powf(beta1, decay));
		v_hat_kernel = vkernel / (1 - powf(beta2, decay));
		m_hat_bias = mbias / (1 - powf(beta1, decay));
		v_hat_bias = vbias / (1 - powf(beta2, decay));

		kernel = kernel.array() - lr * m_hat_kernel.array()/(v_hat_kernel.array().sqrt() + epsilon);
		bias = bias.array() - lr * m_hat_bias.array()/(v_hat_bias.array().sqrt() + epsilon);
	}

	void Conv2d::zero_grad()
	{
		delta.setZero();
		dkernel.setZero();
		mkernel.setZero();
		vkernel.setZero();
		m_hat_kernel.setZero();
		v_hat_kernel.setZero();
		dbias.setZero();
		mbias.setZero();
		vbias.setZero();
		m_hat_bias.setZero();
		v_hat_bias.setZero();
	}

	vector<int> Conv2d::output_shape() { return { batch, oc, oh, ow }; }
}
