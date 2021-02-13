// Copyright 2021 by Jon Dart. All Rights Reserved
#ifndef _UTIL_H
#define _UTIL_H

#ifdef _MSC_VER
// assume little-endian
static uint16_t bswap16(uint16_t x) { return (x); }
static uint32_t bswap32(uint32_t x) { return (x); }
static uint64_t bswap64(uint64_t x) { return (x); }
#elif __BYTE_ORDER == __BIG_ENDIAN
static uint16_t bswap16(uint16_t x) { return __builtin_bswap16(x); }
static uint32_t bswap32(uint32_t x) { return __builtin_bswap32(x); }
static uint64_t bswap64(uint64_t x) { return __builtin_bswap64(x); }
#else
static uint16_t bswap16(uint16_t x) { return (x); }
static uint32_t bswap32(uint32_t x) { return (x); }
static uint64_t bswap64(uint64_t x) { return (x); }
#endif

template <typename T> 
T read_little_endian(std::istream &s)
{
    char buf[sizeof(T)];
    s.read(buf, sizeof(T));
    T input = *(reinterpret_cast<T*>(buf));
    switch(sizeof(T)){
    case 1:
        return static_cast<T>(input);
    case 2: 
        return static_cast<T>(bswap16(input));
    case 4: 
        return static_cast<T>(bswap32(input));
    case 8: 
        return static_cast<T>(bswap64(input));
    default:
        throw std::invalid_argument("unsupported size");
    }
}

#endif
