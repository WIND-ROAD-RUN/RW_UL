#include"dsl_TimeBasedCache_t.hpp"


namespace dsl_TimeBasedCache
{

	TEST(TimeDouble,insert)
	{
		rw::dsl::TimeBasedCache<double, double> cache(50);
		cache.insert(123, 123);
		cache.insert(124, 123);
		cache.insert(125, 123);
		cache.insert(126, 123);
		cache.insert(127, 123);

		ASSERT_EQ(cache.size(), 5);
	}

	TEST(TimeDouble, query)
	{
		rw::dsl::TimeBasedCache<double, double> cache(50);
		cache.insert(120, 120);
		cache.insert(121, 121);
		cache.insert(122, 122);
		cache.insert(123, 123);
		cache.insert(124, 124);
		cache.insert(125, 125);
		cache.insert(126, 126);
		cache.insert(127, 127);
		cache.insert(128, 1278);

		auto result=cache.query(125, 2, true, true);
		std::vector<double> standard{123,124};
		ASSERT_EQ(result, standard);


		auto result1 = cache.query(125, 2, false, true);
		standard={ 126,127 };
		ASSERT_EQ(result1, standard);

		auto result2 = cache.query(125, 2, false, false);
		standard = { 127,126 };
		ASSERT_EQ(result2, standard);

		auto result3 = cache.query(125, 2, true, false);
		standard = { 124,123 };
		ASSERT_EQ(result3, standard);
	}

}
