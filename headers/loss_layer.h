#pragma once
#include "common.h"

namespace simple_nn
{
	class Loss
	{
	public:
		int batch;
		int n_label;
	public:
		Loss() : batch(0), n_label(0) {}

		void set_layer(const vector<int>& input_shape)
		{
			assert(input_shape.size() == 2 && "Loss::set(const vector<int>): Output layer must be linear.");
			batch = input_shape[0];
			n_label = input_shape[1];
		}

		virtual float calc_loss(const MatXf& prev_out, const MatXf& labels, MatXf& prev_delta) = 0;
	};

	class MSELoss : public Loss
	{
	public:
		MSELoss() : Loss() {}

		float calc_loss(const MatXf& prev_out, const MatXf& labels, MatXf& prev_delta) override
		{
			prev_delta.resize(batch, n_label);
			float loss_batch = 0.f, out = 0.f, target = 0.f, loss = 0.f;
			for (int n = 0; n < batch; n++) {
				float running_loss = 0.f;
				for (int i = 0; i < n_label; i++) {
					prev_delta(n, i) = (prev_out(n, i) - labels(n, i));
					loss = prev_out(n, i) - labels(n, i);
					running_loss += loss * loss;
				}
				running_loss /= n_label;
				loss_batch += running_loss;
			}
			return loss_batch / batch;
		}
	};

	class MAELoss : public Loss
	{
	public:
		MAELoss() : Loss() {}

		float calc_loss(const MatXf& prev_out, const MatXf& labels, MatXf& prev_delta) override
		{
			prev_delta.resize(batch, n_label);
			float loss_batch = 0.f, loss = 0.f;
			for (int n = 0; n < batch; n++) {
				float running_loss = 0.f;
				for (int i = 0; i < n_label; i++) {
					(prev_out(n, i) >= labels(n, i)) ? prev_delta(n, i) = prev_out(n, i) * 1.f:
							 	 	 	 	 	 	   prev_delta(n, i) = prev_out(n, i) * -1.f;
					loss = prev_out(n, i) - labels(n, i);
					running_loss += abs(loss);
				}
				running_loss /= n_label;
				loss_batch += running_loss;
			}
			return loss_batch / batch;
		}
	};


	class CrossEntropyLoss : public Loss
	{
	public:
		CrossEntropyLoss() : Loss() {}

		float calc_loss(const MatXf& prev_out, const VecXf& labels, MatXf& prev_delta)
		{
			float loss_batch = 0.f;
			prev_delta = prev_out;
			for (int n = 0; n < batch; n++) {
				int answer_idx = labels[n];
				prev_delta(n, answer_idx) -= 1.f;
				loss_batch -= std::log(prev_out(n, answer_idx));
			}
			return loss_batch / batch;
		}
	};
}
