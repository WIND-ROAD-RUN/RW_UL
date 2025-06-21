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
	// 2to16£¬16to2
	TEST_F(ActivationBitsConvert_T,a)
	{
		for (int i=0;i<1000;i++)
		{
			generate();
			auto num = testObj;
			
			testObj = rw::cla::ActivationBitsConvert::switchBinaryTOHex(testObj);
			testObj = rw::cla::ActivationBitsConvert::switchHexTOBinary(testObj);
			ASSERT_EQ(num, testObj);
		}
	}

	// 2to8,8to2
	TEST_F(ActivationBitsConvert_T, b)
	{
		for (int i = 0; i < 1000; i++)
		{
			generate();
			auto num = testObj;

			testObj = rw::cla::ActivationBitsConvert::switchBinaryTOOct(testObj);
			testObj = rw::cla::ActivationBitsConvert::switchOctTOBinary(testObj, num.size());
			ASSERT_EQ(num, testObj);
		}
	}

	// 2to16,16to8,8to2
	TEST_F(ActivationBitsConvert_T, c)
	{
		for (int i = 0; i < 1000; i++)
		{
			generate();
			auto num = testObj;
			
			testObj = rw::cla::ActivationBitsConvert::switchBinaryTOHex(testObj);
			testObj = rw::cla::ActivationBitsConvert::switchHexTOOct(testObj);
			testObj = rw::cla::ActivationBitsConvert::switchOctTOBinary(testObj, num.size());
			ASSERT_EQ(num, testObj);
		}
	}

	// 2to8,8to16,16to2
	TEST_F(ActivationBitsConvert_T, d)
	{
		for (int i = 0; i < 1000; i++)
		{
			generate();
			auto num = testObj;

			testObj = rw::cla::ActivationBitsConvert::switchBinaryTOOct(testObj);
			testObj = rw::cla::ActivationBitsConvert::switchOctTOHex(testObj);
			testObj = rw::cla::ActivationBitsConvert::switchHexTOBinary(testObj,num.size());
			ASSERT_EQ(num, testObj);
		}
	}

	// 2to16,16to8,8to16,16to8,8to2
	TEST_F(ActivationBitsConvert_T, e)
	{
		for (int i = 0; i < 1000; i++)
		{
			generate();
			auto num = testObj;

			testObj = rw::cla::ActivationBitsConvert::switchBinaryTOHex(testObj);
			testObj = rw::cla::ActivationBitsConvert::switchHexTOOct(testObj);
			testObj = rw::cla::ActivationBitsConvert::switchOctTOHex(testObj);
			testObj = rw::cla::ActivationBitsConvert::switchHexTOOct(testObj);
			testObj = rw::cla::ActivationBitsConvert::switchOctTOBinary(testObj,num.size());
			ASSERT_EQ(num, testObj);
		}
	}

	// 2to8,8to2,2to16,16to8,8to16,16to2
	TEST_F(ActivationBitsConvert_T, f)
	{
		for (int i = 0; i < 1000; i++)
		{
			generate();
			auto num = testObj;

			testObj = rw::cla::ActivationBitsConvert::switchBinaryTOOct(testObj);
			testObj = rw::cla::ActivationBitsConvert::switchOctTOBinary(testObj,num.size());
			testObj = rw::cla::ActivationBitsConvert::switchBinaryTOHex(testObj);
			testObj = rw::cla::ActivationBitsConvert::switchHexTOOct(testObj);
			testObj = rw::cla::ActivationBitsConvert::switchOctTOHex(testObj);
			testObj = rw::cla::ActivationBitsConvert::switchHexTOBinary(testObj,num.size());
			ASSERT_EQ(num, testObj);
		}
	}
}
