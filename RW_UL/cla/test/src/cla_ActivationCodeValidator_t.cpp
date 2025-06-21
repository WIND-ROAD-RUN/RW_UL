#include"cla_ActivationCodeValidator_t.hpp"

#include "ActivationCodeModule/cla_ActivationCodeGenerator.hpp"

namespace cla_ActivationCodeModule
{
	TEST(ActivationCodeValidator, validateActivationCode)
	{
		rw::cla::ActivationCodeGenerator generator;
		auto code=generator.generateActivationCode("temp");
		rw::cla::ActivationCodeValidator validator;
		auto result=validator.validateActivationCode(code, "temp");
		ASSERT_EQ(result, true);
	}
}

