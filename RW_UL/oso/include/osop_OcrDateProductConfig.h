#ifndef OSOP_OCRDATEPRODUCTCONFIG_H_
#define OSOP_OCRDATEPRODUCTCONFIG_H_

#include"osoc_MVSCameraConfig.h"
#include"oso_core.h"

#include<string>

namespace rw {
	namespace oso {
		class ObjectStoreAssembly;
		class ObjectStoreItem;
	}
}

namespace rw {
	namespace oso {
		struct RecognizeRange {
		public:
			std::pair<double, double> leftTopCorner{ 0,0 };
			std::pair<double, double> leftLowerCorner{ 0,0 };
			std::pair<double, double> rightTopCorner{ 0,0 };
			std::pair<double, double> rightLowerCorner{ 0,0 };
		public:
			static ObjectStoreAssembly toObjectStoreAssembly(const RecognizeRange& recognizeRange);
			static RecognizeRange toRecognizeRange(const ObjectStoreAssembly& assembly);

		public:
			RecognizeRange() = default;
			RecognizeRange(const ObjectStoreAssembly& assembly);
			RecognizeRange(std::shared_ptr<ObjectStoreAssembly> assembly);
			operator ObjectStoreAssembly()const;
		};

		struct ProductCheckCount {
		public:
			int totalCount{ 0 };
			int passCount{ 0 };
			int outCount{ 0 };
		public:
			static ObjectStoreAssembly toObjectStoreAssembly(const ProductCheckCount& productCheckCount);
			static ProductCheckCount toProductCheckCount(const ObjectStoreAssembly& assembly);

		public:
			ProductCheckCount() = default;
			ProductCheckCount(const ObjectStoreAssembly& assembly);
			ProductCheckCount(std::shared_ptr<ObjectStoreAssembly> assembly);
			operator ObjectStoreAssembly()const;
		};

		struct RejectAttribute {
		public:
			int RejectDelay{ 0 };
			int OffsetsNumber{ 0 };
			int DisposalTime{ 0 };
		public:
			static ObjectStoreAssembly toObjectStoreAssembly(const RejectAttribute& rejectAttribute);
			static RejectAttribute toRejectAttribute(const ObjectStoreAssembly& assembly);
		public:
			RejectAttribute() = default;
			RejectAttribute(const ObjectStoreAssembly& assembly);
			RejectAttribute(std::shared_ptr<ObjectStoreAssembly> assembly);
			operator ObjectStoreAssembly()const;
		};

		struct OcrDataProductConfig {
		public:
			std::string productName{ "XXXX" };

			CameraAttributesBasic cameraAttributesBasic;
			RecognizeRange recognizeRange;
			ProductCheckCount productCheckCount;
			RejectAttribute rejectAttribute;
		public:
			static ObjectStoreAssembly toObjectStoreAssembly(const OcrDataProductConfig& ocrDataProductConfig);
			static OcrDataProductConfig toOcrDataProductConfig(const ObjectStoreAssembly& assembly);
		public:
			OcrDataProductConfig() = default;
			OcrDataProductConfig(const ObjectStoreAssembly& assembly);
			OcrDataProductConfig(std::shared_ptr<ObjectStoreAssembly> assembly);
			operator ObjectStoreAssembly()const;
		};
	}
}

#endif // OSOP_OCRDATEPRODUCTCONFIG_H_
