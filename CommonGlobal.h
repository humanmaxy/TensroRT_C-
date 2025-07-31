#ifndef __COMMON_GLOBAL_H__
#define __COMMON_GLOBAL_H__

#if defined(COMMONLAYER_LIB)
#define COMMONLAYER_EXPORT __declspec(dllexport)
#else
#define COMMONLAYER_EXPORT __declspec(dllimport)
#endif

#endif // __COMMON_GLOBAL_H__
