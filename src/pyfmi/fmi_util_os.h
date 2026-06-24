#ifndef FMI_UTIL_OS_H
#define FMI_UTIL_OS_H

#include <stdio.h>

#if defined(_WIN32)
    #define os_fseek _fseeki64
    #define os_ftell _ftelli64
#else
    #define os_fseek fseeko
    #define os_ftell ftello
#endif

#endif
