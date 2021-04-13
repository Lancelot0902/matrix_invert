/*
 * Copyright (c) 2020 by Xinsheng Dou.
 * All rights reserved
 *
 * Module for computing the inverse of input matrix
 *
 */

#include "invert.h"
#include <limits>

namespace mcv
{

bool invert(cv::InputArray _src, cv::OutputArray _dst, int method)
{
    bool result = false;
    cv::Mat src = _src.getMat();
    int type = src.type();

    CV_Assert(type == CV_32F || type == CV_64F);

    size_t esz = CV_ELEM_SIZE(type);
    int m = src.rows, n = src.cols;

    CV_Assert(m == n);
    CV_Assert(method == cv::DECOMP_LU || method == cv::DECOMP_CHOLESKY);

    _dst.create(n, n, type);
    cv::Mat dst = _dst.getMat();
    int elem_size = CV_ELEM_SIZE(type);
    cv::AutoBuffer<uchar> buf(n * n * elem_size);
    cv::Mat src1(n, n, type, buf.data());
    src.copyTo(src1);
    setIdentity(dst);

    if (method == cv::DECOMP_LU && type == CV_32F)
        result = mcv::GaussJordan(src1.ptr<float>(), src1.step, n, dst.ptr<float>(), dst.step, n) != 0;
    return result;
}

template <typename _Tp>
static int GaussJordan(_Tp *A, size_t astep, int m, _Tp *b, size_t bstep, int n)
{
    int i, j, k, p = 1;
    astep /= sizeof(A[0]);
    bstep /= sizeof(b[0]);

    for (i = 0; i < m; i++)
    {
        k = i;

        for (j = i + 1; j < m; j++)
            if (std::abs(A[j * astep + i]) > std::abs(A[k * astep + i]))
                k = j;

        if (std::abs(A[k * astep + i]) < EPS)
            return 0;

        if (k != i)
        {
            for (j = i; j < m; j++)
                std::swap(A[i * astep + j], A[k * astep + j]);
            if (b)
                for (j = 0; j < n; j++)
                    std::swap(b[i * bstep + j], b[k * bstep + j]);
            p = -p;
        }

        _Tp d = -1 / A[i * astep + i];

        for (j = i + 1; j < m; j++)
        {
            _Tp alpha = A[j * astep + i] * d;

            for (k = i + 1; k < m; k++)
                A[j * astep + k] += alpha * A[i * astep + k];

            if (b)
                for (k = 0; k < n; k++)
                    b[j * bstep + k] += alpha * b[i * bstep + k];
        }
    }

    if (b)
    {
        for (i = m - 1; i >= 0; i--)
            for (j = 0; j < n; j++)
            {
                _Tp s = b[i * bstep + j];
                for (k = i + 1; k < m; k++)
                    s -= A[i * astep + k] * b[k * bstep + j];
                b[i * bstep + j] = s / A[i * astep + i];
            }
    }

    return p;
}

bool inverse(cv::InputArray _src, int method)
{
    bool result = false;
    cv::Mat src = _src.getMat();
    int type = src.type();

    CV_Assert(type == CV_32F || type == CV_64F);

    size_t esz = CV_ELEM_SIZE(type);
    int m = src.rows, n = src.cols;

    CV_Assert(m == n);

    if (method == cv::DECOMP_LU && type == CV_32F)
        result = mcv::LU(src.ptr<float>(), src.step, n) != 0;
    return result;
}

template <typename T>
static int LU(T *A, size_t astep, int n)
{
    int i, j, k, row, index;
    astep /= sizeof(A[0]);

    int row_start = 0;
    for (i = 0; i != n; ++i)
    {
        k = i;
        T base = A[row_start + i];
        T max = base;

        index = (i + 1) * astep + i;
        for (j = i + 1; j < n; ++j) {
            if (std::abs(A[index]) > std::abs(max)) {
                k = j;
                max = A[index];
            }
            index += astep;
        }

        if (std::abs(max) < EPS)
            return 0;

        T d = 1 / base;

        int alpha = max / base;
        // T coef = 1 / base * alpha; // useless!!!

        // row = row_start;

        for (j = 0; j < n; ++j)
            A[row_start + j] *= alpha;

        A[row_start + i] = d;

        int row_start2 = 0;

        for (j = 0; j < n; ++j)
        {
            if (i == j) {
                row_start2 += astep;
                continue;
            }
            // row = j * astep;
            d = -A[row_start2 + i];
            A[row_start2 + i] = 0;

// #pragma omp parallel for
            for (k = 0; k < n; ++k)
                A[row_start2 + k] += A[row_start + k] * d;

            row_start2 += astep;
        }
        row_start += astep;
    }
    return 1;
}

} // namespace mcv
