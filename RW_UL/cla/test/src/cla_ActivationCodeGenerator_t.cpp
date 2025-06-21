#include"cla_ActivationCodeGenerator_t.hpp"

namespace cla_ActivationCodeGenerator
{

	TEST(ActivationCodeGenerator, construct)
	{
		rw::cla::ActivationCodeGenerator generator;
		std::cout <<generator.generateActivationCode("asdawda");
	}



}