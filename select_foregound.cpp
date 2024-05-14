#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

using namespace cv;

struct g_didtr_params
{
	double tau, mu[3], cov[9];
};

struct g_didtr_params2
{
	double c, mu[3], u[9];
};

inline double prob(const double* p_src, g_didtr_params2* p)
{
	double v[3];
	for (int i = 0; i < 3; i++)
	{
		v[i] = p_src[i] - p->mu[i];
	}

	double s = p->c;
	double* w = p->u;
	for (int i = 0; i < 3; i++)
	{
		double d = v[0] * w[0] + v[1] * w[1] + v[2] * w[2];
		s += d * d;
		w += 3;
	}
	return s;
}

// N - foregound distributions, K - all distributions.

template <int N, int K>
void calc_probability(const uchar* p_src, uchar* p_dst, int cnt, const g_didtr_params* d)
{
	float exp_t[256];

	for (int i = 0; i < 256; i++)
		exp_t[i] = exp(-i / 16.0);

	g_didtr_params2 distr[K];

	for (int i = 0; i < K; i++)
	{
		Mat1d cova = Mat1d(3, 3, (double*)d[i].cov) * 255. * 255.;
		distr[i].c = -log(d[i].tau / (pow(2 * CV_PI, 3.0 / 2.0) * sqrt(determinant(cova))));
		for (int j = 0; j < 3; j++)
		{
			distr[i].mu[j] = 255. * d[i].mu[j];
		}

		cv::Mat1d w, u, vt;
		cv::SVD::compute(cova, w, u, vt, cv::SVD::FULL_UV);

		double* p = w.ptr<double>();
		for (int j = 0; j < 3; j++)
		{
			p[j] = 1.0 / sqrt(2 * p[j]);
		}

		cv::Mat1d c = cv::Mat::diag(w) * u.t();
		memcpy(distr[i].u, c.data, sizeof(distr[i].u));
	}

	for (int i = 0; i < cnt; i++)
	{
		double data[] = { p_src[0], p_src[1], p_src[2] };
		double s[K];
		for (int j = 0; j < K; j++)
			s[j] = prob(data, distr + j);

		double s_min = s[0];
		for (int j = 1; j < K; j++)
			if (s[j] < s_min)
				s_min = s[j];

		for (int j = 0; j < K; j++)
		{
			double v = s[j] - s_min;
			s[j] = v < 16 ? exp_t[(int)(v * 16)] : 0;
		}

		double s_fg = 0, s_bg = 0;
		for (int j = 0; j < N; j++)
			s_fg += s[j];
		for (int j = N; j < K; j++)
			s_bg += s[j];

		*p_dst++ = 255.0 * s_fg / (s_fg + s_bg);
		p_src += 3;
	}
}

cv::Mat1b select_foreground(cv::Mat3b& img)
{
	static const g_didtr_params c[] =
	{ {
	1.913211,
	{0.972325,0.725950,0.602071},
	{0.000672,0.000488,0.000094,0.000488,0.000700,0.000528,0.000094,0.000528,0.000942}
	},{
	0.123472,
	{0.995186,0.784121,0.679268},
	{0.000053,-0.000086,-0.000122,-0.000086,0.003334,0.003693,-0.000122,0.003693,0.004475}
	},{
	0.049128,
	{0.832681,0.388465,0.143306},
	{0.012360,0.014595,0.008144,0.014595,0.024270,0.024143,0.008144,0.024143,0.036578}
	},{
	0.150872,
	{0.331887,0.272338,0.658897},
	{0.036815,-0.026004,-0.005714,-0.026004,0.062413,0.025112,-0.005714,0.025112,0.023568}
	} };

	cv::Mat1b bg(img.size());
	calc_probability<2, 4>(img.ptr(), bg.ptr(), img.total(), c);
	return bg;
}
