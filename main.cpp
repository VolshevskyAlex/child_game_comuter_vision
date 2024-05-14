#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <filesystem>

#include "match_pict.h"

using namespace cv;
using namespace std;
namespace fs = std::filesystem;

void split_jpg(string fn, Mat3b &img, vector<Mat3b> &pict_sampl, Mat3b &v)
{
	cv::Mat3b img0 = cv::imread(fn);
	cv::resize(img0, img, cv::Size(), .3, .3);
	split_imgs(img, pict_sampl, v);
}

void match_imgs_in_dir()
{
	std::vector<std::string> a;
	fs::path img_path = "../imgs";
	try {
		for (const auto& entry : fs::directory_iterator(img_path))
		{
			a.push_back(entry.path().string());
		}
	}
	catch (std::exception& e)
	{
		std::cout << e.what() << std::endl;
		return;
	}

	Matcher matcher;
	MatchInfo info;

	int fr_no = 0;
	int fps = 10;
	cv::Size frame_sz(1920, 1080);
//	cv::VideoWriter vw("out.mp4", VideoWriter::fourcc('H', '2', '6', '4'), fps, frame_sz);
	for (;;)
	{
		int ix0 = rand() % a.size(), ix1;
		do {
			ix1 = rand() % a.size();
		} while (ix1 == ix0);

		std::string fn0 = a[ix0];
		std::string fn1 = a[ix1];
		printf("n=%d\t%d\t%d\t%s\t%s\n", fr_no++, ix0, ix1, fn0.c_str(), fn1.c_str());

		cv::Mat3b img0, img1, v0, v1;
		vector<Mat3b> pict_sampl0, pict_sampl1;
		split_jpg(fn0, img0, pict_sampl0, v0);
		split_jpg(fn1, img1, pict_sampl1, v1);
		matcher.match_pict_list(pict_sampl0, pict_sampl1, info);

		cv::Mat3b img_a = pict_sampl0[info.i];
		cv::Mat3b img_b = pict_sampl1[info.j];
		cv::Matx23d a = cv::getRotationMatrix2D(cv::Point2f(img_b.cols / 2, img_b.rows / 2), info.angle, info.scale);
		a(0, 2) += info.center.x;
		a(1, 2) += info.center.y;

		cv::Mat3b img_b_rt, diff;
		cv::warpAffine(img_b, img_b_rt, a, img_a.size(), INTER_LINEAR, BORDER_CONSTANT, Scalar::all(255));

		cv::Mat3b v2 = draw_pict(pict_sampl0);
		cv::Mat3b v3 = draw_pict(pict_sampl1);

		int d = 2;
		cv::Mat3b v_frame(cv::Size((v2.cols + d) * 2 + 200, v2.rows + d + img0.rows + d));
		v_frame.setTo(0);
		cv::Point p(0, 0);
		img0.copyTo(v_frame(Rect(p.x, p.y, img0.cols, img0.rows)));
		p.x += img0.cols + d;

		img_a.copyTo(v_frame(Rect(p.x, p.y, img_a.cols, img_a.rows)));
		p.y += img_a.rows + d;
		img_b_rt.copyTo(v_frame(Rect(p.x, p.y, img_b_rt.cols, img_b_rt.rows)));

		p.y = 0;
		p.x += img_b_rt.cols + d;
		img1.copyTo(v_frame(Rect(p.x, p.y, img1.cols, img1.rows)));

		p.x = 0;
		p.y = img0.rows + 1;
		v2.copyTo(v_frame(Rect(p.x, p.y, v2.cols, v2.rows)));
		p.x += v2.cols + d;
		v3.copyTo(v_frame(Rect(p.x, p.y, v2.cols, v2.rows)));
		p.x = 0;
		p.y = v2.rows + d;
		cv::Mat3b v_frame2;

		double scale = (double)frame_sz.height / v_frame.rows;
		cv::resize(v_frame, v_frame2, cv::Size(), scale, scale);

		v_frame2 = v_frame2(Rect(0, 0, frame_sz.width, frame_sz.height));

//		for (int j = 0; j < fps; j++)
//			vw << v_frame2;

		imshow("v_frame2", v_frame2);

		if (waitKey(100) == 27)
			break;
	}
}

int main()
{
	match_imgs_in_dir();
}
