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

		template < typename ImageType>
		struct ElementInfo
		{
		public:
			ImageType image;
			QMap<Name, Value> attribute;
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

			inline ElementInfo< ImageType> getImage(const KeyType& history)
			{
				auto result = imageCache.get(history);
				if (!result.has_value())
				{
					return ElementInfo<ImageType>();
				}
				return result.value();
			}

			inline void setImage(const KeyType& history, const ElementInfo<ImageType>& imageInfo)
			{
				imageCache.set(history, imageInfo);
			}
		};

	}
}
