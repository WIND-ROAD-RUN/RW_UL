#include"cla_utility_t.hpp"

#include "cla_utility.hpp"
#include <gtest/gtest.h>

#include"cryptopp/cryptlib.h"
#include"cryptopp/filters.h"
#include"cryptopp/modes.h"
#include"cryptopp/aes.h"
#include"cryptopp/hex.h"

namespace cla_utility
{
	TEST(a,b)
	{
		// 测试输入
		std::string input = "Hello, Crypto++!";
		std::string key = "1234567890123456"; 
		std::string iv = "6543210987654321"; 

		// 加密结果
		std::string encrypted;

		try
		{
			// 使用Crypto++进行AES加密
			CryptoPP::AES::Encryption aesEncryption(reinterpret_cast<const unsigned char*>(key.data()), key.size());
			CryptoPP::CBC_Mode_ExternalCipher::Encryption cbcEncryption(aesEncryption, reinterpret_cast<const unsigned char*>(iv.data()));

			CryptoPP::StringSource ss(input, true,
				new CryptoPP::StreamTransformationFilter(cbcEncryption,
					new CryptoPP::StringSink(encrypted)));

			// 将加密结果转换为十六进制字符串
			std::string hexEncoded;
			CryptoPP::StringSource hexSource(encrypted, true,
				new CryptoPP::HexEncoder(
					new CryptoPP::StringSink(hexEncoded)));

			// 输出加密结果
			std::cout << "Encrypted (Hex): " << hexEncoded << std::endl;

			// 检查加密结果是否非空
			ASSERT_FALSE(encrypted.empty());
		}
		catch (const CryptoPP::Exception& e)
		{
			// 如果加密失败，测试失败
			FAIL() << "Encryption failed: " << e.what();
		}
	}


}
