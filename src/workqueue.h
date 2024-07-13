#pragma once
#include "taggedptr.h"
#include "soa.h"

#include <thrust/tuple.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>

template <typename WorkItem>
class WorkQueue:public SOA<WorkItem>
{
private:
    cuda::atomic<int, cuda::thread_scope_device> size{0};
public:
    WorkQueue() = default;
    WorkQueue(int n, Allocator alloc) : SOA<WorkItem>(n, alloc) {}
    __host__ __device__ int Size() const
    {
        return size.load(cuda::std::memory_order_acquire);
    }
    __host__ __device__ void Reset()
    {
        size.store(0, cuda::std::memory_order_release);
    }
    __host__ __device__ int Push(WorkItem w)
    {
        int index = AllocateEntry();
        (*this)[index] = w;
        return index;
    }
protected:
    __host__ __device__ int AllocateEntry()
    {
        return size.fetch_add(1, cuda::std::memory_order_release);
    }

};

template <typename F>
__global__ void Kernel(F func, int nItems) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nItems)
        return;

    func(tid);
}

template <typename F>
void GPUParallelFor(int n, int blockSize, F func)
{
    auto kernel = &Kernel<F>;
    int gridSize = (n + blockSize - 1) / blockSize;
    kernel<<<gridSize, blockSize>>>(func, nItems);
}

template <typename F, typename WorkItem>
void ForAllQueued(const WorkQueue<WorkItem> *q, int maxQueued, int blockSize, F&& func)
{
    GPUParallelFor(maxQueued, blockSize, [=] __device__ (int index) mutable {
        if (index >= q->Size())
            return;
        func((*q)[index]);
    });
}


template <typename... Ts>
class MultiWorkQueue {
    using Types = TypePack<WorkQueue<Ts>...>;
public:
    // MultiWorkQueue Public Methods
    template <typename T>
    __host__ __device__ WorkQueue<T> *Get() {
        using Tp = typename std::remove_cv_t<T>;
        return &thrust::get<IndexOf<WorkQueue<T>, Types>::count>(queues);
    }

    MultiWorkQueue(int n, Allocator alloc, bool* haveType) {
        int index = 0;
        ((thrust::get<IndexOf<WorkQueue<T>, Types>::count>(queues) = WorkQueue<Ts>(haveType[index++] ? n : 1, alloc)), ...);
    }

    template <typename T>
    __host__ __device__ int Size() const {
        return Get<T>()->Size();
    }

    template <typename T>
    __host__ __device__ int Push(const T &value) {
        return Get<T>()->Push(value);
    }

    __host__ __device__ void Reset() {
        (Get<Ts>()->Reset(), ...);
    }

private:
    // MultiWorkQueue Private Members
    thrust::tuple<WorkQueue<Ts>...> queues;
};
