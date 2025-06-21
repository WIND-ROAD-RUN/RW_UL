#include"cla_utility.hpp"

#include <iomanip>
#include <sstream>
#include <vector>
#include <bitset>

namespace rw
{
	namespace cla
	{
		ActivationString ActivationBitsConvert::switchBinaryTOHex(const ActivationString& str)
		{
			// 检查输入是否为有效的二进制字符串
			if (str.empty() || str.find_first_not_of("01") != std::string::npos)
			{
				throw std::invalid_argument("输入的字符串不是有效的二进制字符串");
			}

			// 将二进制字符串分割为每 8 位一组
			std::ostringstream hexStream;
			for (size_t i = 0; i < str.size(); i += 8)
			{
				// 提取 8 位二进制子串
				std::string byteStr = str.substr(i, 8);

				// 将二进制子串转换为整数
				unsigned char byte = static_cast<unsigned char>(std::bitset<8>(byteStr).to_ulong());

				// 将整数转换为十六进制字符串
				hexStream << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(byte);
			}

			// 返回十六进制字符串
			return hexStream.str();
		}

		ActivationString ActivationBitsConvert::switchHexTOBinary(const ActivationString& str)
		{
			if (str.empty() || str.find_first_not_of("0123456789ABCDEFabcdef") != std::string::npos)
			{
				throw std::invalid_argument("输入的字符串不是有效的十六进制字符串");
			}

			std::ostringstream binaryStream;
			for (char hexChar : str)
			{
				unsigned char byte = static_cast<unsigned char>(std::stoi(std::string(1, hexChar), nullptr, 16));

				binaryStream << std::bitset<4>(byte);
			}

			return binaryStream.str();
		}

		ActivationString ActivationBitsConvert::switchOctTOBinary(const ActivationString& str, size_t size )
		{
			// 检查输入是否为有效的八进制字符串
			if (str.empty() || str.find_first_not_of("01234567") != std::string::npos)
			{
				throw std::invalid_argument("输入的字符串不是有效的八进制字符串");
			}

			// 将八进制字符串转换为二进制字符串
			std::ostringstream binaryStream;
			for (char octChar : str)
			{
				// 将每个八进制字符转换为整数
				unsigned char byte = static_cast<unsigned char>(std::stoi(std::string(1, octChar), nullptr, 8));

				// 使用 std::bitset 将整数转换为 3 位二进制字符串
				binaryStream << std::bitset<3>(byte);
			}

			// 返回生成的二进制字符串
			return binaryStream.str();
		}

		ActivationString ActivationBitsConvert::switchOctTOBinary(const ActivationString& str)
		{

		}

		ActivationString ActivationBitsConvert::switchBinaryTOOct(const ActivationString& str)
		{
			// 检查输入是否为有效的二进制字符串
			if (str.empty() || str.find_first_not_of("01") != std::string::npos)
			{
				throw std::invalid_argument("输入的字符串不是有效的二进制字符串");
			}

			// 将二进制字符串分割为每 3 位一组
			std::ostringstream octStream;
			for (size_t i = 0; i < str.size(); i += 3)
			{
				// 提取 3 位二进制子串
				std::string bitStr = str.substr(i, 3);

				// 如果不足 3 位，补充 0
				while (bitStr.size() < 3)
				{
					bitStr = "0" + bitStr;
				}

				// 将二进制子串转换为整数
				unsigned char byte = static_cast<unsigned char>(std::bitset<3>(bitStr).to_ulong());

				// 将整数转换为八进制字符
				octStream << static_cast<int>(byte);
			}

			// 返回生成的八进制字符串
			return octStream.str();
		}

		ActivationString ActivationBitsConvert::switchOctTOHex(const ActivationString& str)
		{
			// 检查输入是否为有效的八进制字符串
			if (str.empty() || str.find_first_not_of("01234567") != std::string::npos)
			{
				throw std::invalid_argument("输入的字符串不是有效的八进制字符串");
			}

			// 将八进制字符串转换为二进制字符串
			ActivationString binaryStr = switchOctTOBinary(str);

			// 将二进制字符串转换为十六进制字符串
			ActivationString hexStr = switchBinaryTOHex(binaryStr);

			// 返回生成的十六进制字符串
			return hexStr;
		}

		ActivationString ActivationBitsConvert::switchHexTOOct(const ActivationString& str)
		{
			// 检查输入是否为有效的十六进制字符串
			if (str.empty() || str.find_first_not_of("0123456789ABCDEFabcdef") != std::string::npos)
			{
				throw std::invalid_argument("输入的字符串不是有效的十六进制字符串");
			}

			// 将十六进制字符串转换为二进制字符串
			ActivationString binaryStr = switchHexTOBinary(str);

			// 将二进制字符串转换为八进制字符串
			ActivationString octStr = switchBinaryTOOct(binaryStr);

			// 返回生成的八进制字符串
			return octStr;
		}
	}
}
