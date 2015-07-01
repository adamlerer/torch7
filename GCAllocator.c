/**
 * TH allocator that automatically triggers garbage collection
 * based on a dynamic heap soft max, as well as on malloc failure.
 */

#include "GCAllocator.h"
#include "general.h"

#if defined(TH_DISABLE_HEAP_TRACKING)
#elif (defined(__unix) || defined(_WIN32))
#include <malloc.h>
#elif defined(__APPLE__)
#include <malloc/malloc.h>
#endif

typedef struct {
  void (*collectGarbage)(void *gcData);
  void *gcData;
  long torchHeapSize;
  long torchHeapSizeSoftMax;
} GCAllocatorData;

static long getAllocSize(void *ptr) {
#if defined(TH_DISABLE_HEAP_TRACKING)
  return 0;
#elif defined(__unix)
  return malloc_usable_size(ptr);
#elif defined(__APPLE__)
  return malloc_size(ptr);
#elif defined(_WIN32)
  return _msize(ptr);
#else
  return 0;
#endif
}

/* (1) if the torch-allocated heap size exceeds the soft max, run GC
 * (2) if post-GC heap size exceeds 80% of the soft max, increase the
 *     soft max by 40%
 */
static void maybeTriggerGC(GCAllocatorData *allocator) {
  if(allocator->torchHeapSize > allocator->torchHeapSizeSoftMax) {
    allocator->collectGarbage(allocator->gcData);
    if(allocator->torchHeapSize > allocator->torchHeapSizeSoftMax * 0.8) {
      allocator->torchHeapSizeSoftMax = allocator->torchHeapSize * 1.4; // FIXME: torchHeapSizeSoftMax
    }
  }
}

static void heapIncr(void *ptr, GCAllocatorData *allocator) {
  allocator->torchHeapSize += getAllocSize(ptr);
}
static void heapDecr(void *ptr, GCAllocatorData *allocator) {
  allocator->torchHeapSize -= getAllocSize(ptr);
}

static void* GCAllocInternal(long size, GCAllocatorData *allocator)
{
  void *ptr;

  // duplicate code from lib/TH/THGeneral.c; if you change this, change that
  if (size > 5120)
  {
#if (defined(__unix) || defined(__APPLE__)) && (!defined(DISABLE_POSIX_MEMALIGN))
    if (posix_memalign(&ptr, 64, size) != 0)
      ptr = NULL;
#elif defined(_WIN32)
    ptr = _aligned_malloc(size, 64);
#else
    ptr = malloc(size);
#endif
  }
  else
  {
    ptr = malloc(size);
  }

  heapIncr(ptr, allocator);
  return ptr;
}

static void* GCAlloc(long size, void *_allocator)
{
  GCAllocatorData *allocator = _allocator;
  void *ptr;

  if(size < 0)
    THError("$ Torch: invalid memory size -- maybe an overflow?");

  if(size == 0)
    return NULL;

  ptr = GCAllocInternal(size, allocator);

  if(!ptr) {
    allocator->collectGarbage(allocator->gcData);
    ptr = GCAllocInternal(size, allocator);
  }

  if(!ptr)
    THError("$ Torch: not enough memory: you tried to allocate %dGB. Buy new RAM!", size/1073741824);

  maybeTriggerGC(allocator);
  return ptr;
}

static void* GCRealloc(void *ptr, long size, void *_allocator)
{
  GCAllocatorData *allocator = _allocator;
  if(!ptr)
    return(THAlloc(size));

  if(size == 0)
  {
    THFree(ptr);
    return NULL;
  }

  if(size < 0)
    THError("$ Torch: invalid memory size -- maybe an overflow?");

  heapDecr(ptr, allocator);
  void *newptr = realloc(ptr, size);

  if(!newptr) {
    allocator->collectGarbage(allocator->gcData);
    newptr = realloc(ptr, size);
  }
  heapIncr(newptr ? newptr : ptr, allocator);

  if(!newptr)
    THError("$ Torch: not enough memory: you tried to reallocate %dGB. Buy new RAM!", size/1073741824);

  maybeTriggerGC(allocator);
  return newptr;
}

static void GCFree(void *ptr, void *_allocator)
{
  GCAllocatorData *allocator = _allocator;
  heapDecr(ptr, allocator);
  free(ptr);
}


void THUseGCAllocator(void (*collectGarbageFunction)(void *gcData), void *gcData) {
  GCAllocatorData *allocator = malloc(sizeof(GCAllocatorData));
  allocator->collectGarbage = collectGarbageFunction;
  allocator->gcData = gcData;
  allocator->torchHeapSize = 0;
  allocator->torchHeapSizeSoftMax = 30000000; // 300MB, adjusted upward dynamically
  THSetAllocator(GCAlloc, GCRealloc, GCFree, allocator);
}
