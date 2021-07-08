#include <vector>
#include <array>
#include "MurmurHash3.h"


//Credit to findingprotopia.org blog for original struct
template<class Key>
class BloomFilter
{
private:
  using filter = std::vector<bool>;
  filter m_bits;
  uint8_t m_numHashes;



public:

  //initializer list for filter
  BloomFilter(uint64_t size, uint8_t numHashes) : m_bits(size), m_numHashes(numHashes) {}


  //'unique' hash functions - check out the paper on two hash
  // bloom filters to learn more.
  uint64_t nthHash (uint8_t n, uint64_t first_hash, uint64_t second_hash, uint64_t filterSize){

    return (first_hash  +  n*second_hash) % filterSize;

  }

  std::array<uint64_t, 2> hash(const Key * data){

    //hash the key

    std::array<uint64_t, 2> hashValue;

    //test replacement with x86 -  dereference pointer to get num bytes
    MurmurHash3_x64_128(data, sizeof(Key), 0, hashValue.data());

    return hashValue;

  }

  void add(const Key * data){

    auto hashValues = hash(data);

    for (int n=0; n < m_numHashes;  n++){
      m_bits[nthHash(n, hashValues[0], hashValues[1], m_bits.size())] = true;
    }

  }

  bool possiblyContains(const Key* data){

    auto hashValues  = hash(data);

    for (int n=0;  n < m_numHashes; n++){
      if (!m_bits[nthHash(n,hashValues[0], hashValues[1], m_bits.size())]){
        return false;
      }
    }
    return true;
  }
};
