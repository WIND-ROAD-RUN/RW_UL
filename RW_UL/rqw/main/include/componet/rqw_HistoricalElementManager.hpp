#pragma once

#include<QImage>
#include <QDateTime>

#include"dsl_CacheFIFO.hpp"

namespace rw
{
	namespace rqw
	{
		template <typename ElementType, typename Name = QString, typename Value = double>
		struct ElementInfo
		{
		public:
			ElementType element;
			QMap<Name, Value> attribute;
		public:
			ElementInfo(const ElementType& img) : element(img) {}
			ElementInfo() = default;
		};

		template <typename KeyType, typename ImageType>
		class HistoricalElementManager
		{
		private:
			rw::dsl::CacheFIFO<KeyType, ElementInfo< ImageType>> elementCache;
		public:
			HistoricalElementManager(size_t capacity = 100)
				: elementCache(capacity)
			{
			}

			inline void insertElement(const KeyType& history, const ElementInfo< ImageType>& imageInfo)
			{
				elementCache.set(history, imageInfo);
			}

			inline std::optional<ElementInfo<ImageType>> getElement(const KeyType& history)
			{
				auto result = elementCache.get(history);

				return result;
			}

			inline void setElement(const KeyType& history, const ElementInfo<ImageType>& imageInfo)
			{
				elementCache.set(history, imageInfo);
			}
		};

	}
}
