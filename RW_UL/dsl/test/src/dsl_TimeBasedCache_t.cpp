#include"dsl_TimeBasedCache_t.hpp"

#include <chrono>


namespace dsl_TimeBasedCache
{

	TEST(TimeDouble, insert)
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
		cache.insert(128, 128);

		auto result = cache.query(125, 127, true, true,true);
		std::vector<double> standard{ 125,126,127 };
		ASSERT_EQ(result, standard);

		result = cache.query(125, 127, false, true, true);
		standard = {126,127 };
		ASSERT_EQ(result, standard);

		result = cache.query(125, 127, false, false, true);
		standard = { 126 };
		ASSERT_EQ(result, standard);

		result = cache.query(125, 127, true, false, true);
		standard = { 125,126 };
		ASSERT_EQ(result, standard);

		result = cache.query(125, 127, true, true, false);
		standard = { 127,126,125 };
		ASSERT_EQ(result, standard);
	}

	TEST(TimeDouble, newQuery)
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
		cache.insert(128, 128);

		auto result = cache.query(125, 2, true, true);
		std::vector<double> standard{ 123,124 };
		ASSERT_EQ(result, standard);


		auto result1 = cache.query(125, 2, false, true);
		standard = { 126,127 };
		ASSERT_EQ(result1, standard);

		auto result2 = cache.query(125, 2, false, false);
		standard = { 127,126 };
		ASSERT_EQ(result2, standard);

		auto result3 = cache.query(125, 2, true, false);
		standard = { 124,123 };
		ASSERT_EQ(result3, standard);
	}

	TEST(Timetime, newQuery)
	{
		using Time = std::chrono::system_clock::time_point;
		rw::dsl::TimeBasedCache<Time, double> cache(50);

		// 构造一组递增的时间点
		Time base = std::chrono::system_clock::now();
		std::vector<Time> times;
		for (int i = 0; i < 9; ++i) {
			times.push_back(base + std::chrono::seconds(i));
		}

		cache.insert(times[0], 120);
		cache.insert(times[1], 121);
		cache.insert(times[2], 122);
		cache.insert(times[3], 123);
		cache.insert(times[4], 124);
		cache.insert(times[5], 125);
		cache.insert(times[6], 126);
		cache.insert(times[7], 127);
		cache.insert(times[8], 128);

		// 查询时用 times[5] 代表“125”
		auto result = cache.queryWithTime(times[5], 2, true, true);
		std::vector<double> standard{ 123, 124 };
		ASSERT_EQ(result, standard);

		auto result1 = cache.queryWithTime(times[5], 2, false, true);
		standard = { 126, 127 };
		ASSERT_EQ(result1, standard);

		auto result2 = cache.queryWithTime(times[5], 2, false, false);
		standard = { 127, 126 };
		ASSERT_EQ(result2, standard);

		auto result3 = cache.queryWithTime(times[5], 2, true, false);
		standard = { 124, 123 };
		ASSERT_EQ(result3, standard);
	}

	TEST(TimeDouble, queryWithTime)
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
		cache.insert(128, 128);

		auto result = cache.queryWithTime(125, 2, true, true);
		std::vector<double> standard = { 124,125 };
		ASSERT_EQ(result, standard);

		auto result1 = cache.queryWithTime(125, 2, true, false);
		standard = { 125,124 };
		ASSERT_EQ(result1, standard);

		auto result2 = cache.queryWithTime(125, 2, false, true);
		standard = { 125,126 };
		ASSERT_EQ(result2, standard);

		auto result3 = cache.queryWithTime(125, 2, false, false);
		standard = { 126,125 };
		ASSERT_EQ(result3, standard);
	}

	TEST(TimeDouble, queryToMap)
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
		cache.insert(128, 128);

		auto result = cache.queryToMap(125, 2, true, true);
		std::unordered_map<double,double> standard = {{ 123,123},{124,124} };
		ASSERT_EQ(result, standard);

		auto result1 = cache.queryToMap(125, 2, true, false);
		standard = { {124,124},{123,123}};
		ASSERT_EQ(result1, standard);

		auto result2 = cache.queryToMap(125, 2, false, true);
		standard = { {126,126},{127,127} };
		ASSERT_EQ(result2, standard);

		auto result3 = cache.queryToMap(125, 2, false, false);
		standard = { {127,127},{126,126}};
		ASSERT_EQ(result3, standard);
	}

	TEST(TimeDouble, queryWithTimeToMap)
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
		cache.insert(128, 128);

		auto result = cache.queryWithTimeToMap(125, 2, true, true);
		std::unordered_map<double, double> standard = { { 124,124},{125,125} };
		ASSERT_EQ(result, standard);

		auto result1 = cache.queryWithTimeToMap(125, 2, true, false);
		standard = { {125,125},{124,124} };
		ASSERT_EQ(result1, standard);

		auto result2 = cache.queryWithTimeToMap(125, 2, false, true);
		standard = { {125,125},{126,126} };
		ASSERT_EQ(result2, standard);

		auto result3 = cache.queryWithTimeToMap(125, 2, false, false);
		standard = { {126,126},{125,125} };
		ASSERT_EQ(result3, standard);
	}

	TEST(TimeDouble, size)
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
		cache.insert(128, 128);

		auto result = cache.query(125, 2, true, true);
		ASSERT_EQ(result.size(), 2);

		result = cache.queryWithTime(125, 2, true, true);
		ASSERT_EQ(result.size(), 2);
	}

	TEST(TimeDouble, insertTime)
	{
		using KeyType = std::chrono::system_clock::time_point; 
		rw::dsl::TimeBasedCache<KeyType, double> cache(50);

		for (int i = 0; i < 30; i++) {
			KeyType key = std::chrono::system_clock::now(); 
			cache.insert(key, i);
		}

		ASSERT_EQ(cache.size(), 30);
	}
}
