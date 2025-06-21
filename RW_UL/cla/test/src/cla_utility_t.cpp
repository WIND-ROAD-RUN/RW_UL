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
		for (int i=0;i<1000;i++)
		{
			generate();
			auto num = testObj;
			//// 2to16
			//testObj = rw::cla::ActivationBitsConvert::switchBinaryTOHex(testObj);
			//testObj = rw::cla::ActivationBitsConvert::switchHexTOBinary(testObj);
			//ASSERT_EQ(num, testObj);

			// 2to8
			testObj = rw::cla::ActivationBitsConvert::switchBinaryTOOct(testObj);
			testObj = rw::cla::ActivationBitsConvert::switchOctTOBinary(testObj, testObj.size());
			ASSERT_EQ(num, testObj);
		}
	}


}
