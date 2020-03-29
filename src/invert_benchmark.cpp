#include <benchmark/benchmark.h>
#include <opencv2/core.hpp>
#include <iostream>
#include "invert/invert.h"

static void custom_args(benchmark::internal::Benchmark* b)
{
    for (int i = 100; i <= 1000; i += 100) {
        b->Arg(i);
    }
}

template<typename T>
static void BM_CV_INVERT(benchmark::State &state)
{
    int size = state.range(0);
    cv::Mat m(size, size, CV_32F, cv::Scalar(0));
    cv::Mat n(size, size, CV_32F, cv::Scalar(0));

    cv::randu(m, cv::Scalar::all(0), cv::Scalar::all(255));

    for (auto _ : state)
    {
        cv::invert(m, n);
    }
    state.SetItemsProcessed(state.iterations() * 1);
}

#define RUN_CV_INVERT_FUNCTIONS(type) BENCHMARK_TEMPLATE(BM_CV_INVERT,float)->RangeMultiplier(10)->Apply(custom_args);

RUN_CV_INVERT_FUNCTIONS(float);

template<typename T>
static void BM_MCV_INVERT(benchmark::State &state)
{
    int size = state.range(0);
    cv::Mat m(size, size, CV_32F, cv::Scalar(0));
    cv::randu(m, cv::Scalar::all(0), cv::Scalar::all(255));

    for (auto _ : state)
    {
        mcv::inverse(m,cv::DECOMP_LU);
    }
    state.SetItemsProcessed(state.iterations() * 1);
}

#define RUN_MCV_INVERT_FUNCTIONS(type) BENCHMARK_TEMPLATE(BM_MCV_INVERT,float)->RangeMultiplier(10)->Apply(custom_args);

RUN_MCV_INVERT_FUNCTIONS(float);

BENCHMARK_MAIN();
