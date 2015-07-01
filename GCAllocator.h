#ifndef TORCH_GCALLOCATOR_INC
#define TORCH_GCALLOCATOR_INC

void THUseGCAllocator(void (*collectGarbageFunction)(void *gcData), void *gcData);

#endif
