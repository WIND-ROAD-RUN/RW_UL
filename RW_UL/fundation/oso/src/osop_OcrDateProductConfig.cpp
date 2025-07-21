#include"osop_OcrDateProductConfig.h"

namespace rw
{
	namespace oso {
		ObjectStoreAssembly
			RecognizeRange::toObjectStoreAssembly
			(const RecognizeRange& recognizeRange)
		{
			ObjectStoreAssembly assembly;
			assembly.setName("RecognizeRange");

			{
				ObjectStoreItem leftTopCorner_x;
				leftTopCorner_x.setName("LeftTopCorner_x");
				leftTopCorner_x.setValueFromDouble(recognizeRange.leftTopCorner.first);
				assembly.addItem(makeObjectStoreItemSharedPtr(leftTopCorner_x));

				ObjectStoreItem leftTopCorner_y;
				leftTopCorner_y.setName("LeftTopCorner_y");
				leftTopCorner_y.setValueFromDouble(recognizeRange.leftTopCorner.second);
				assembly.addItem(makeObjectStoreItemSharedPtr(leftTopCorner_y));
			}

			{
				ObjectStoreItem leftLowerCorner_x;
				leftLowerCorner_x.setName("LeftLowerCorner_x");
				leftLowerCorner_x.setValueFromDouble(recognizeRange.leftLowerCorner.first);
				assembly.addItem(makeObjectStoreItemSharedPtr(leftLowerCorner_x));

				ObjectStoreItem leftLowerCorner_y;
				leftLowerCorner_y.setName("LeftLowerCorner_y");
				leftLowerCorner_y.setValueFromDouble(recognizeRange.leftLowerCorner.second);
				assembly.addItem(makeObjectStoreItemSharedPtr(leftLowerCorner_y));
			}

			{
				ObjectStoreItem rightTopCorner_x;
				rightTopCorner_x.setName("RightTopCorner_x");
				rightTopCorner_x.setValueFromDouble(recognizeRange.rightTopCorner.first);
				assembly.addItem(makeObjectStoreItemSharedPtr(rightTopCorner_x));

				ObjectStoreItem rightTopCorner_y;
				rightTopCorner_y.setName("RightTopCorner_y");
				rightTopCorner_y.setValueFromDouble(recognizeRange.rightTopCorner.second);
				assembly.addItem(makeObjectStoreItemSharedPtr(rightTopCorner_y));
			}

			{
				ObjectStoreItem rightLowerCorner_x;
				rightLowerCorner_x.setName("RightLowerCorner_x");
				rightLowerCorner_x.setValueFromDouble(recognizeRange.rightLowerCorner.first);
				assembly.addItem(makeObjectStoreItemSharedPtr(rightLowerCorner_x));

				ObjectStoreItem rightLowerCorner_y;
				rightLowerCorner_y.setName("RightLowerCorner_y");
				rightLowerCorner_y.setValueFromDouble(recognizeRange.rightLowerCorner.second);
				assembly.addItem(makeObjectStoreItemSharedPtr(rightLowerCorner_y));
			}

			return assembly;
		}

		RecognizeRange
			RecognizeRange::toRecognizeRange
			(const ObjectStoreAssembly& assembly)
		{
			RecognizeRange recognizeRange;
			for (auto& item : assembly.getItems())
			{
				auto nItem = ObjectStoreCoreToItem(item);
				if (nItem->getName() == "LeftTopCorner_x")
				{
					recognizeRange.leftTopCorner.first = nItem->getValueAsDouble();
				}
				else if (nItem->getName() == "LeftTopCorner_y")
				{
					recognizeRange.leftTopCorner.second = nItem->getValueAsDouble();
				}
				else if (nItem->getName() == "LeftLowerCorner_x")
				{
					recognizeRange.leftLowerCorner.first = nItem->getValueAsDouble();
				}
				else if (nItem->getName() == "LeftLowerCorner_y")
				{
					recognizeRange.leftLowerCorner.second = nItem->getValueAsDouble();
				}
				else if (nItem->getName() == "RightTopCorner_x")
				{
					recognizeRange.rightTopCorner.first = nItem->getValueAsDouble();
				}
				else if (nItem->getName() == "RightTopCorner_y")
				{
					recognizeRange.rightTopCorner.second = nItem->getValueAsDouble();
				}
				else if (nItem->getName() == "RightLowerCorner_x")
				{
					recognizeRange.rightLowerCorner.first = nItem->getValueAsDouble();
				}
				else if (nItem->getName() == "RightLowerCorner_y")
				{
					recognizeRange.rightLowerCorner.second = nItem->getValueAsDouble();
				}
			}

			return recognizeRange;
		}

		RecognizeRange::RecognizeRange(const ObjectStoreAssembly& assembly)
		{
			*this = toRecognizeRange(assembly);
		}

		RecognizeRange::RecognizeRange(std::shared_ptr<ObjectStoreAssembly> assembly)
		{
			*this = toRecognizeRange(*assembly);
		}

		RecognizeRange::operator ObjectStoreAssembly() const
		{
			return toObjectStoreAssembly(*this);
		}

		ObjectStoreAssembly
			ProductCheckCount::toObjectStoreAssembly
			(const ProductCheckCount& productCheckCount)
		{
			ObjectStoreAssembly assembly;
			assembly.setName("ProductCheckCount");

			{
				ObjectStoreItem totalCount;
				totalCount.setName("TotalCount");
				totalCount.setValueFromInt(productCheckCount.totalCount);
				assembly.addItem(makeObjectStoreItemSharedPtr(totalCount));
			}

			{
				ObjectStoreItem passCount;
				passCount.setName("PassCount");
				passCount.setValueFromInt(productCheckCount.passCount);
				assembly.addItem(makeObjectStoreItemSharedPtr(passCount));
			}

			{
				ObjectStoreItem outCount;
				outCount.setName("OutCount");
				outCount.setValueFromInt(productCheckCount.outCount);
				assembly.addItem(makeObjectStoreItemSharedPtr(outCount));
			}

			return assembly;
		}

		ProductCheckCount
			ProductCheckCount::toProductCheckCount
			(const ObjectStoreAssembly& assembly)
		{
			ProductCheckCount productCheckCount;
			for (auto& item : assembly.getItems())
			{
				auto nItem = ObjectStoreCoreToItem(item);
				if (nItem->getName() == "TotalCount")
				{
					productCheckCount.totalCount = nItem->getValueAsInt();
				}
				else if (nItem->getName() == "PassCount")
				{
					productCheckCount.passCount = nItem->getValueAsInt();
				}
				else if (nItem->getName() == "OutCount")
				{
					productCheckCount.outCount = nItem->getValueAsInt();
				}
			}

			return productCheckCount;
		}

		ProductCheckCount::ProductCheckCount(const ObjectStoreAssembly& assembly)
		{
			*this = toProductCheckCount(assembly);
		}

		ProductCheckCount::ProductCheckCount(std::shared_ptr<ObjectStoreAssembly> assembly)
		{
			*this = toProductCheckCount(*assembly);
		}

		ProductCheckCount::operator ObjectStoreAssembly() const
		{
			return toObjectStoreAssembly(*this);
		}

		ObjectStoreAssembly
			RejectAttribute::toObjectStoreAssembly
			(const RejectAttribute& rejectAttribute)
		{
			ObjectStoreAssembly assembly;
			assembly.setName("RejectAttribute");

			{
				ObjectStoreItem RejectDelay;
				RejectDelay.setName("RejectDelay");
				RejectDelay.setValueFromInt(rejectAttribute.RejectDelay);
				assembly.addItem(makeObjectStoreItemSharedPtr(RejectDelay));
			}

			{
				ObjectStoreItem OffsetsNumber;
				OffsetsNumber.setName("OffsetsNumber");
				OffsetsNumber.setValueFromInt(rejectAttribute.OffsetsNumber);
				assembly.addItem(makeObjectStoreItemSharedPtr(OffsetsNumber));
			}

			{
				ObjectStoreItem DisposalTime;
				DisposalTime.setName("DisposalTime");
				DisposalTime.setValueFromInt(rejectAttribute.DisposalTime);
				assembly.addItem(makeObjectStoreItemSharedPtr(DisposalTime));
			}

			return assembly;
		}

		RejectAttribute
			RejectAttribute::toRejectAttribute
			(const ObjectStoreAssembly& assembly)
		{
			RejectAttribute rejectAttribute;

			for (auto& item : assembly.getItems())
			{
				auto nItem = ObjectStoreCoreToItem(item);
				if (nItem->getName() == "RejectDelay")
				{
					rejectAttribute.RejectDelay = nItem->getValueAsInt();
				}
				else if (nItem->getName() == "OffsetsNumber")
				{
					rejectAttribute.OffsetsNumber = nItem->getValueAsInt();
				}
				else if (nItem->getName() == "DisposalTime")
				{
					rejectAttribute.DisposalTime = nItem->getValueAsInt();
				}
			}

			return rejectAttribute;
		}

		RejectAttribute::RejectAttribute(const ObjectStoreAssembly& assembly)
		{
			*this = toRejectAttribute(assembly);
		}

		RejectAttribute::RejectAttribute(std::shared_ptr<ObjectStoreAssembly> assembly)
		{
			*this = toRejectAttribute(*assembly);
		}

		RejectAttribute::operator ObjectStoreAssembly() const
		{
			return toObjectStoreAssembly(*this);
		}

		ObjectStoreAssembly
			OcrDataProductConfig::toObjectStoreAssembly
			(const OcrDataProductConfig& ocrDataProductConfig)
		{
			ObjectStoreAssembly assembly;

			assembly.setName("OcrDataProductConfig");

			{
				ObjectStoreItem productName;
				productName.setName("ProductName");
				productName.setValueFromString(ocrDataProductConfig.productName);
				assembly.addItem(makeObjectStoreItemSharedPtr(productName));
			}

			{
				ObjectStoreAssembly cameraAttributesBasic = CameraAttributesBasic::toObjectStoreAssembly(ocrDataProductConfig.cameraAttributesBasic);
				cameraAttributesBasic.setName("CameraAttributesBasic");
				assembly.addItem(makeObjectStoreAssemblySharedPtr(cameraAttributesBasic));
			}

			{
				ObjectStoreAssembly recognizeRange = RecognizeRange::toObjectStoreAssembly(ocrDataProductConfig.recognizeRange);
				recognizeRange.setName("RecognizeRange");
				assembly.addItem(makeObjectStoreAssemblySharedPtr(recognizeRange));
			}

			{
				ObjectStoreAssembly productCheckCount = ProductCheckCount::toObjectStoreAssembly(ocrDataProductConfig.productCheckCount);
				productCheckCount.setName("ProductCheckCount");
				assembly.addItem(makeObjectStoreAssemblySharedPtr(productCheckCount));
			}

			{
				ObjectStoreAssembly rejectAttribute = RejectAttribute::toObjectStoreAssembly(ocrDataProductConfig.rejectAttribute);
				rejectAttribute.setName("RejectAttribute");
				assembly.addItem(makeObjectStoreAssemblySharedPtr(rejectAttribute));
			}

			return assembly;
		}

		OcrDataProductConfig
			OcrDataProductConfig::toOcrDataProductConfig
			(const ObjectStoreAssembly& assembly)
		{
			OcrDataProductConfig ocrDataProductConfig;

			for (auto& item : assembly.getItems())
			{
				auto itemType = item->getStoreType();
				if (itemType == std::string("assembly")) {
					auto nItem = ObjectStoreCoreToAssembly(item);
					if (nItem->getName() == "CameraAttributesBasic")
					{
						ocrDataProductConfig.cameraAttributesBasic = CameraAttributesBasic::toCameraAttributesBasic(*nItem);
					}
					else if (nItem->getName() == "RecognizeRange")
					{
						ocrDataProductConfig.recognizeRange = RecognizeRange::toRecognizeRange(*nItem);
					}
					else if (nItem->getName() == "ProductCheckCount")
					{
						ocrDataProductConfig.productCheckCount = ProductCheckCount::toProductCheckCount(*nItem);
					}
					else if (nItem->getName() == "RejectAttribute")
					{
						ocrDataProductConfig.rejectAttribute = RejectAttribute::toRejectAttribute(*nItem);
					}
					else
					{
						static_assert(true, "Unknown item type");
					}
				}
				else if (itemType == std::string("item")) {
					auto nItem = ObjectStoreCoreToItem(item);
					if (nItem->getName() == "ProductName")
					{
						ocrDataProductConfig.productName = nItem->getValueAsString();
					}
					else {
						static_assert(true, "Unknown item type");
					}
				}
			}

			return ocrDataProductConfig;
		}

		OcrDataProductConfig::OcrDataProductConfig(const ObjectStoreAssembly& assembly)
		{
			*this = toOcrDataProductConfig(assembly);
		}

		OcrDataProductConfig::OcrDataProductConfig(std::shared_ptr<ObjectStoreAssembly> assembly)
		{
			*this = toOcrDataProductConfig(*assembly);
		}

		OcrDataProductConfig::operator ObjectStoreAssembly() const
		{
			return toObjectStoreAssembly(*this);
		}
	}
}