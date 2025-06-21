#include"cla_utility_t.hpp"

#include "cla_utility.hpp"
#include <gtest/gtest.h>

#include"cryptopp/cryptlib.h"
#include"cryptopp/filters.h"
#include"cryptopp/modes.h"
#include"cryptopp/aes.h"
#include"cryptopp/hex.h"

namespace cla_ActivationBitsConvert
{
	TEST_F(ActivationBitsConvert_T,b)
	{
		for (int i=0;i<100;i++)
		{
			generate();
			std::cout << testObj << std::endl;
		}
	}


}
