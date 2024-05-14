#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <numeric>

#include "match_pict.h"

using namespace cv;

const int pict_size = 200;

Matcher::Matcher()
{
	cv::Size sz(fft_size_, fft_size_);
	Point2f center(sz.width / 2, sz.height / 2);
	double d = hypot(center.x, center.y);
	log_base_ = pow(10.0, log10(d) / sz.width);
	map_x_.create(sz);
	map_y_.create(sz);
	for (int i = 0; i < sz.height; i++)
	{
		double theta = CV_PI * (0.5 + (double)i / sz.height);
		double sin_theta = sin(theta);
		double cos_theta = cos(theta);
		for (int j = 0; j < sz.width; j++)
		{
			double radius = pow(log_base_, (double)j);
			double x = radius * sin_theta + center.x;
			double y = radius * cos_theta + center.y;
			map_x_(i, j) = (float)x;
			map_y_(i, j) = (float)y;
		}
	}
	highpass_f_ = create_gauss_FFT(6, 1).mul(create_highpass_FFT());
}

void Matcher::fft_shift(cv::Mat1f& a)
{
	cv::Mat1f tmp(a.size());
	int w = a.cols / 2, h = a.rows / 2;
	a(Rect(0, 0, w, h)).copyTo(tmp(Rect(w, h, w, h)));
	a(Rect(w, h, w, h)).copyTo(tmp(Rect(0, 0, w, h)));
	a(Rect(w, 0, w, h)).copyTo(tmp(Rect(0, h, w, h)));
	a(Rect(0, h, w, h)).copyTo(tmp(Rect(w, 0, w, h)));
	a = tmp;
}

cv::Mat1f Matcher::create_gauss_FFT(int radius, double sigma)
{
	cv::Size imsize(fft_size_, fft_size_);
	Mat kernelX = getGaussianKernel(2 * radius + 1, sigma, CV_32F);
	Mat kernelY = getGaussianKernel(2 * radius + 1, sigma, CV_32F);
	// create 2d gaus
	Mat1f kernel = kernelX * kernelY.t();

	int w = imsize.width - kernel.cols;
	int h = imsize.height - kernel.rows;

	int r = w / 2;
	int l = imsize.width - kernel.cols - r;

	int b = h / 2;
	int t = imsize.height - kernel.rows - b;

	copyMakeBorder(kernel, kernel, t, b, l, r, BORDER_CONSTANT, Scalar::all(0));
	fft_shift(kernel);

	cv::Mat1f F0[2];
	Mat complexImg;
	dft(kernel, complexImg, DFT_COMPLEX_OUTPUT);
	split(complexImg, F0);
	fft_shift(F0[0]);

	return F0[0];
}

cv::Mat1f Matcher::create_highpass_FFT()
{
	Size sz(fft_size_, fft_size_);
	Mat a = Mat(sz.height, 1, CV_32FC1);
	Mat b = Mat(1, sz.width, CV_32FC1);

	float step_y = CV_PI / sz.height;
	float val = -CV_PI * 0.5;

	for (int i = 0; i < sz.height; ++i)
	{
		a.at<float>(i) = cos(val);
		val += step_y;
	}

	val = -CV_PI * 0.5;
	float step_x = CV_PI / sz.width;
	for (int i = 0; i < sz.width; ++i)
	{
		b.at<float>(i) = cos(val);
		val += step_x;
	}

	Mat tmp = a * b;
	tmp = (1.0 - tmp).mul(2.0 - tmp);

	return tmp;
}

void Matcher::to_fft_log_polar(const cv::Mat3b& img, cv::Mat1f& img_f, cv::Mat1f& lp_FFT)
{
	const int dw = fft_size_ - img.cols;
	const int dh = fft_size_ - img.cols;
	Mat1f F0[2];
	Mat1f f0;
	cv::Mat1b im0;
	cv::cvtColor(img, im0, cv::COLOR_BGR2GRAY);
	cv::copyMakeBorder(im0, im0, dw - dw/2, dh - dh/2, dw/2, dh/2, BORDER_CONSTANT, Scalar::all(255));
	im0.convertTo(img_f, CV_32FC1, 1.0 / 255.0);

	Mat complexImg;
	dft(img_f, complexImg, DFT_COMPLEX_OUTPUT);
	split(complexImg, F0);
	magnitude(F0[0], F0[1], f0);
	fft_shift(f0);

	f0 = f0.mul(highpass_f_);
	cv::normalize(f0, f0, 0, 1, NORM_MINMAX);

	cv::Mat1f magn_lp;
	cv::remap(f0, magn_lp, map_x_, map_y_, cv::INTER_LINEAR, BORDER_CONSTANT, Scalar(0, 0, 0));
	cv::dft(magn_lp, lp_FFT, DFT_REAL_OUTPUT);

//	cv::imshow("img", im0);
//	cv::imshow("magn_lp", magn_lp);
//	cv::waitKey();
}

static Point2d weightedCentroid(cv::Mat1f src, cv::Point peakLocation, cv::Size weightBoxSize, double* response)
{
	int minr = peakLocation.y - (weightBoxSize.height >> 1);
	int maxr = peakLocation.y + (weightBoxSize.height >> 1);
	int minc = peakLocation.x - (weightBoxSize.width >> 1);
	int maxc = peakLocation.x + (weightBoxSize.width >> 1);

	Point2d centroid;
	double sumIntensity = 0.0;

	// clamp the values to min and max if needed.
	if (minr < 0)
	{
		minr = 0;
	}

	if (minc < 0)
	{
		minc = 0;
	}

	if (maxr > src.rows - 1)
	{
		maxr = src.rows - 1;
	}

	if (maxc > src.cols - 1)
	{
		maxc = src.cols - 1;
	}

	const float* dataIn = src.ptr<float>();
	dataIn += minr * src.cols;
	for (int y = minr; y <= maxr; y++)
	{
		for (int x = minc; x <= maxc; x++)
		{
			centroid.x += (double)x * dataIn[x];
			centroid.y += (double)y * dataIn[x];
			sumIntensity += (double)dataIn[x];
		}

		dataIn += src.cols;
	}

	if (response)
		*response = sumIntensity;

	sumIntensity += DBL_EPSILON; // prevent div0 problems...

	centroid.x /= sumIntensity;
	centroid.y /= sumIntensity;

	return centroid;
}

#pragma pack(push, 1)
struct complex_t
{
	float Re;
	float Im;
};
#pragma pack(pop)

static void inline phase_corr_eq(complex_t& a, complex_t& b, complex_t& dst)
{
	float v1 = a.Re * b.Re + a.Im * b.Im;
	float v2 = a.Im * b.Re - a.Re * b.Im;
	float mag = hypotf(v1, v2);

	if (mag==0)
	{
		dst.Re = 0;
		dst.Im = 0;
	}
	else
	{
		dst.Re = v1 / mag;
		dst.Im = v2 / mag;
	}
}

static void inline phase_corr_eq_f(float* a, float* b, float* dst, int step)
{
	complex_t t_a{ a[0], a[step] }, t_b{ b[0], b[step]}, t_c;
	phase_corr_eq(t_a, t_b, t_c);
	dst[0] = t_c.Re;
	dst[step] = t_c.Im;
}

void perf_phase_corr_eq(cv::Mat1f& FFT1, cv::Mat1f& FFT2, cv::Mat1f& C)
{
	// CCS packed format (complex-conjugate-symmetrical)
	int M = FFT1.rows;
	int N = FFT1.cols;
	int K = N / 2 - 1;
	C.create(M, N);

	float* f_a = FFT1.ptr<float>(0, 0);
	float* f_b = FFT2.ptr<float>(0, 0);
	float* f_c = C.ptr<float>(0, 0);

	complex_t* p_a = (complex_t*)(f_a + 1);
	complex_t* p_b = (complex_t*)(f_b + 1);
	complex_t* p_c = (complex_t*)(f_c + 1);
	for (int i = 0; i < M; i++, p_a++, p_b++, p_c++)
	{
		for (int j = 0; j < K; j++)
		{
			phase_corr_eq(*p_a++, *p_b++, *p_c++);
		}
	}

	int ofs = N;
	for (int i = 1; i < M / 2; i++)
	{
		phase_corr_eq_f(f_a + ofs, f_b + ofs, f_c + ofs, N);
		ofs += N - 1;
		phase_corr_eq_f(f_a + ofs, f_b + ofs, f_c + ofs, N);
		ofs += N + 1;
	}
	C.at<float>(0, 0) = 1.f;
	C.at<float>(0, C.cols-1) = 1.f;
	C.at<float>(C.rows - 1, 0) = 1.f;
	C.at<float>(C.rows - 1, C.cols - 1) = 1.f;
}

cv::Point2d Matcher::phase_correlate_custom(cv::Mat1f& FFT1, cv::Mat1f& FFT2, double &response)
{
	cv::Mat1f C;

	perf_phase_corr_eq(FFT1, FFT2, C);

	idft(C, C);

	fft_shift(C); // shift the energy to the center of the frame.

	// locate the highest peak
	Point peakLoc;
	minMaxLoc(C, NULL, NULL, NULL, &peakLoc);

	// get the phase shift with sub-pixel accuracy, 5x5 window seems about right here...
	Point2d t;
	t = weightedCentroid(C, peakLoc, Size(5, 5), &response);

	// max response is M*N (not exactly, might be slightly larger due to rounding errors)

	int M = FFT1.rows;
	int N = FFT1.cols;

	response /= M * N;

	// adjust shift relative to image center...
	Point2d center((double)FFT1.cols / 2.0, (double)FFT1.rows / 2.0);

	return (center - t);
}

void Matcher::match_pict_list(const  std::vector<cv::Mat3b>& sampl0, const  std::vector<cv::Mat3b>& sampl1, MatchInfo& match)
{
	const int M = sampl0.size();
	const int N = sampl1.size();
	std::vector<cv::Mat1f> imgs_a(M), logpolar_a(M), imgs_b(N), logpolar_b(N);
	for (int i = 0; i < M; i++)
	{
		to_fft_log_polar(sampl0[i], imgs_a[i], logpolar_a[i]);
	}
	for (int i = 0; i < N; i++)
	{
		to_fft_log_polar(sampl1[i], imgs_b[i], logpolar_b[i]);
	}

	double a[8 * 8];
	std::vector<A> applic_list;
	for (int i = 0; i < M; i++)
	{
		for (int j = 0; j < N; j++)
		{
			A it;
			it.rotation_and_scale = phase_correlate_custom(logpolar_b[j], logpolar_a[i], it.response);

			it.i = i;
			it.j = j;
			applic_list.push_back(it);
			a[i * 8 + j] = it.response;
		}
	}
	for (int i = 0; i < 8; i++)
	{
		for (int j = 0; j < 8; j++)
		{
			printf("%f\t", a[i * 8 + j]);
		}
		printf("\n");
	}
	std::sort(applic_list.begin(), applic_list.end(), [](A& a, A& b) {return a.response > b.response; });
	for (int i = 0; i < 8; i++)
	{
		printf("%d, %d, %f\n", applic_list[i].i, applic_list[i].j, applic_list[i].response);
	}

	match.i = -1;
	match.j = -1;
	double response, max_response = -DBL_MAX;
	for (size_t i = 0; i < applic_list.size(); i++)
	{
		const A &it = applic_list[i];
		if (it.response < applic_list[0].response * .5)
			break;

		float angle = 180.0 * it.rotation_and_scale.y / fft_size_;
		float scale = pow(log_base_, it.rotation_and_scale.x);
		// --------------
		if (scale > 1.8)
		{
			std::cout << "Images are not compatible. Scale change > 1.8\n";
		}

		for (int j = 0; j < 2; j++)
		{
			// Now rotate and scale fragment back, then find translation
			Matx23d rot_mat = getRotationMatrix2D_(Point(fft_size_ / 2, fft_size_ / 2), angle, 1.0 / scale);

			// rotate and scale
			Mat im1_rs;
			cv::warpAffine(imgs_b[it.j], im1_rs, rot_mat, cv::Size(fft_size_, fft_size_), 1, BORDER_CONSTANT, Scalar::all(1));

			// find translation
			Point2d tr = cv::phaseCorrelate(im1_rs, imgs_a[it.i], cv::noArray(), &response);
			printf("%d\t%d\t%f\n", it.i, it.j, response);
			if (response > max_response)
			{
				max_response = response;
				match.i = it.i;
				match.j = it.j;
				match.response = response;
				match.center = tr;
				match.angle = angle;
				match.scale = 1. / scale;
			}
			angle += 180;
		}
	}
}

void split_imgs(cv::Mat3b img, std::vector<cv::Mat3b>& pict_sampl, cv::Mat3b& v)
{
	cv::Mat1b f = select_foreground(img) > 128;//  fg > 0.5;
	std::vector<std::vector<cv::Point> >  contours;
	cv::findContours(f, contours, cv::RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
	int ix = 0;
	double max_a = 0;
	for (size_t i = 0; i < contours.size(); i++)
	{
		double l = cv::contourArea(contours[i]);
		if (l > max_a)
		{
			ix = i;
			max_a = l;
		}
	}
	cv::Mat1b c = cv::Mat1b::zeros(img.size());

	cv::drawContours(c, contours, ix, cv::Scalar(255), -1);
	cv::drawContours(v, contours, ix, cv::Scalar(255), 1);

	cv::bitwise_not(f, f);
	cv::bitwise_and(c, f, c);
	cv::findContours(c, contours, cv::RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

	std::vector<int> idx(contours.size());
	std::vector<double> c_area(contours.size());
	for (size_t i = 0; i < contours.size(); i++)
	{
		idx[i] = i;
		c_area[i] = cv::contourArea(contours[i]);
	}

	std::sort(idx.begin(), idx.end(), [&](int a, int b) {return c_area[a] > c_area[b]; });

	cv::Mat1b mask(img.size());
	for (size_t i = 0; i < idx.size(); i++)
	{
		std::vector<cv::Point>& c = contours[idx[i]];
		cv::Point p1 = c[0], p2 = c[0];
		for (size_t j = 1; j < c.size(); j++)
		{
			if (c[j].x < p1.x)
				p1.x = c[j].x;
			if (c[j].x > p2.x)
				p2.x = c[j].x;
			if (c[j].y < p1.y)
				p1.y = c[j].y;
			if (c[j].y > p2.y)
				p2.y = c[j].y;
		}
		cv::Rect r(p1.x, p1.y, p2.x - p1.x + 1, p2.y - p1.y + 1);
		mask.setTo(0);
		cv::drawContours(mask, contours, idx[i], cv::Scalar(255), -1);
		cv::drawContours(v, contours, idx[i], cv::Scalar(255), 1);

		cv::Mat3b pict(r.size());
		pict.setTo(255);
		img(r).copyTo(pict, mask(r));
		double scale = (double)pict_size / std::max(pict.rows, pict.cols);
		cv::resize(pict, pict, cv::Size(), scale, scale);

		cv::Mat3b pict_b(pict_size, pict_size);
		pict_b.setTo(255);
		pict.copyTo(pict_b(cv::Rect((pict_size - pict.cols) / 2, (pict_size - pict.rows) / 2, pict.cols, pict.rows)));
		pict_sampl.push_back(pict_b);

		if (i == 7)
			break;
	}
}

cv::Mat3b draw_pict(const std::vector<cv::Mat3b>& pict_sampl)
{
	cv::Mat3b img2(cv::Size(pict_size * 4, pict_size * 2));
	for (size_t i = 0; i < pict_sampl.size(); i++)
	{
		int x = (i % 4) * pict_size;
		int y = (i / 4) * pict_size;
		pict_sampl[i].copyTo(img2(cv::Rect(x, y, pict_size, pict_size)));
	}
	return img2;
}
