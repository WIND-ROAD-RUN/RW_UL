#ifndef OSO_CAMERACONFIG_H_
#define OSO_CAMERACONFIG_H_

#include<string>
#include<memory>

namespace rw {
    namespace oso {
        class ObjectStoreAssembly;
    }
}

namespace rw {
    namespace oso
    {
        struct CameraAttributesBasic
        {
        public:
            long exposureTime{ 10000 };
            int gain{ 1 };
        public:
            static ObjectStoreAssembly toObjectStoreAssembly(const CameraAttributesBasic& cameraAttributesBasic);
            static CameraAttributesBasic toCameraAttributesBasic(const ObjectStoreAssembly& assembly);
        public:
            CameraAttributesBasic() = default;

            CameraAttributesBasic(const ObjectStoreAssembly& assembly);

            CameraAttributesBasic(std::shared_ptr<ObjectStoreAssembly> assembly);

            operator ObjectStoreAssembly()const;
        };

        enum class CameraTriggerMode
        {
            Software,
            Hardware,
            Undefined
        };

        struct CameraAttributesAdvanced
        {
        public:
            CameraTriggerMode mode{ CameraTriggerMode::Undefined };
        public:
            static ObjectStoreAssembly toObjectStoreAssembly(const CameraAttributesAdvanced& cameraAttributesBasic);
            static CameraAttributesAdvanced toCameraAttributesAdvanced(const ObjectStoreAssembly& assembly);
        public:
            CameraAttributesAdvanced() = default;

            CameraAttributesAdvanced(const ObjectStoreAssembly& assembly);

            CameraAttributesAdvanced(std::shared_ptr<ObjectStoreAssembly> assembly);

            operator ObjectStoreAssembly()const;
        };

    }

    
}


#endif // !oso_cameraConfig_H_
