/*
 * Copyright (c) 2020 by Xinsheng Dou.
 * All rights reserved
 *
 * Module for computing the inverse of input matrix
 *
 */

#ifndef INVERT_H
#define INVERT_H

#include <iostream>
#include <opencv2/core.hpp>

#define EPS FLT_EPSILON*10

namespace mcv {
	bool invert(cv::InputArray src, cv::OutputArray dst,int method = cv::DECOMP_LU);
	template<typename T> static int GaussJordan(T *A, size_t astep, int n, T *B, size_t bstep,int m);

	bool inverse(cv::InputArray src, int method = cv::DECOMP_LU);
	template<typename T> static int LU(T *A, size_t astep, int n);
} // namespace mcv

#endif // !INVERT_H
