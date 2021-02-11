// Copyright 2021 by Jon Dart. All Rights Reserved
#ifndef _UTIL_H
#define _UTIL_H

static constexpr std::size_t SIMD_SIZE = 32;

#if __BYTE_ORDER == __BIG_ENDIAN
#ifdef __GNUC__
static uint16_t bswap16(uint16_t x) { return __builtin_bswap16(x); }
static uint32_t bswap32(uint32_t x) { return __builtin_bswap32(x); }
static uint64_t bswap64(uint64_t x) { return __builtin_bswap64(x); }
#endif
#endif

template <typename T> 
T read_little_endian(std::istream &s)
{
    char buf[sizeof(T)];
    s.read(buf, sizeof(T));
    T input = *(reinterpret_cast<T*>(buf));
#if __BYTE_ORDER == __BIG_ENDIAN
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
#else
    return input;
#endif
}

template<typename T>
struct ClassOf {};

template<typename Return, typename Class>
struct ClassOf<Return (Class::*)>{
    using type = Class;
};

template< typename T>
using ClassOf_t = typename ClassOf<T>::type;

#endif
