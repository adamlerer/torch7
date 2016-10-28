#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/THTensorCopy.c"
#else

void THTensor_(copy)(THTensor *dst, THTensor *src)
{
  long srcN = THTensor_(nElement)(src);
  long dstN = THTensor_(nElement)(dst);
  if (srcN != dstN) {
    THError("inconsistent tensor nElement: %ld, %ld",
        srcN, dstN);
  }
  int srcContig = THTensor_(isContiguous)(src);
  int dstContig = THTensor_(isContiguous)(dst);
  int doMemcpy = srcContig && dstContig && (sizeof(real) < sizeof(int));
  if (doMemcpy) {
    real *src_data =  src->storage->data;
    real *dst_data =  dst->storage->data;
    if (src_data != dst_data) {
      memcpy(dst_data, src_data, dstN * sizeof(real));
    }
  } else {
    TH_TENSOR_APPLY2(real, dst, real, src, *dst_data = (real)(*src_data);)
  }
}

#define IMPLEMENT_THTensor_COPY(TYPENAMESRC, TYPE_SRC) \
void THTensor_(copy##TYPENAMESRC)(THTensor *tensor, TH##TYPENAMESRC##Tensor *src) \
{ \
  TH_TENSOR_APPLY2(real, tensor, TYPE_SRC, src, *tensor_data = (real)(*src_data);) \
}

IMPLEMENT_THTensor_COPY(Byte, unsigned char)
IMPLEMENT_THTensor_COPY(Char, char)
IMPLEMENT_THTensor_COPY(Short, short)
IMPLEMENT_THTensor_COPY(Int, int)
IMPLEMENT_THTensor_COPY(Long, long)
IMPLEMENT_THTensor_COPY(Float, float)
IMPLEMENT_THTensor_COPY(Double, double)


#endif
