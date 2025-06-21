#include"ActivationCodeModule/cla_ActivationCodeGenerator.hpp"

#include <openssl/sha.h>
#include <iomanip>
#include <sstream>

namespace rw
{
	namespace cla
	{
		ActivationConfig ActivationCodeGenerator::getActivationConfig()
		{
			return _config;
		}

		void ActivationCodeGenerator::setActivationConfig(const ActivationConfig& config)
		{
			_config = config;
		}

		ActivationCodeGenerator::ActivationCodeGenerator(const ActivationConfig& config)
			:_config(config)
		{
		}

		ActivationString ActivationCodeGenerator::generateActivationCode(const UniqueIdentifier& indetifier)
		{
			// 序列化配置字符串
			auto configStr = ActivationConfig::serialize(_config);

			// 将 UniqueIdentifier 和 configStr 组合
			std::string combinedData = indetifier + configStr;

			// 使用 SHA256 生成哈希值
			unsigned char hash[SHA256_DIGEST_LENGTH];
			SHA256(reinterpret_cast<const unsigned char*>(combinedData.c_str()), combinedData.size(), hash);

			// 将哈希值转换为十六进制字符串
			std::ostringstream activationCodeStream;
			for (unsigned char byte : hash)
			{
				activationCodeStream << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(byte);
			}

			// 返回生成的激活码
			return activationCodeStream.str();

		}
	}
}