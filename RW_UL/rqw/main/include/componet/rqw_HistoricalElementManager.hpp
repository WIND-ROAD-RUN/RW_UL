#pragma once

#include<QImage>
#include <QDateTime>

#include"dsl_CacheFIFO.hpp"

namespace rw
{
	namespace rqw
	{
		using Name = QString;
		using Value = QString;

		template < typename ElementType>
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
			rw::dsl::CacheFIFO<KeyType, ElementInfo< ImageType>> imageCache;
		public:
			HistoricalElementManager(size_t capacity = 100)
				: imageCache(capacity)
			{
			}

			inline void insertImage(const KeyType& history, const ElementInfo< ImageType>& imageInfo)
			{
				imageCache.set(history, imageInfo);
			}

			inline std::optional<ElementInfo<ImageType>> getImage(const KeyType& history)
			{
				auto result = imageCache.get(history);

				return result;
			}

			inline void setImage(const KeyType& history, const ElementInfo<ImageType>& imageInfo)
			{
				imageCache.set(history, imageInfo);
			}
		};

	}
}
