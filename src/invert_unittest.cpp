#include <iostream>
#include <gtest/gtest.h>
#include "invert/invert.h"

bool check_2_matrices(const cv::Mat &matrix1, const cv::Mat &matrix2, float eps)
{
	int m = matrix1.rows;
	int n = matrix1.cols;
	const float *p;
	const float *q;
	bool flag = true;
	float max = 0;
	float a = 0; 
	float b = 0;
	for (int i = 0; i != m; ++i)
	{
		p = matrix1.ptr<float>(i);
		q = matrix2.ptr<float>(i);
		for (int j = 0; j != n; ++j)
		{
			//			std::cout<<p[j]<<" "<<q[j]<<std::endl;
			if (std::fabs(p[j] - q[j]) > eps)
			{
				if (std::fabs(p[j] - q[j]) > std::fabs(max))
				{
					a = p[j];
					b = q[j];
					max = p[j] - q[j];
				}
				flag = false;
			}
		}
	}
	std::cout << a << " " << b << std::endl;
	std::cout << max << std::endl;
	return flag;
}

TEST(invert, example1)
{
	int n = 1000;
	cv::Mat in = cv::Mat(n, n, CV_32F);
	cv::Mat out1(n, n, CV_32F, cv::Scalar(0));
	cv::Mat out2(n, n, CV_32F, cv::Scalar(0));
	cv::randu(in, cv::Scalar::all(-1000), cv::Scalar::all(1000));
	//	std::cout<<in<<std::endl;
	cv::invert(in, out1);
	//	std::cout<<out1<<std::endl;
	mcv::inverse(in);
	//	std::cout<<in<<std::endl;

	bool isEqual = check_2_matrices(out1, in, 1e-5);
	EXPECT_TRUE(isEqual);
}

int main(int argc, char **argv)
{
	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}