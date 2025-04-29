#pragma once

#include<QString>
#include<QVector>

namespace rw {
    namespace hoec
    {
        class CameraPassive;
    }

    namespace rqw {
        struct CameraMetaData
        {
            QString ip{ "Undefined" };
            QString provider{ "Undefined" };
        public:
            CameraMetaData() = default;
            CameraMetaData(QString ip, QString provider)
                :ip(std::move(ip)), provider(std::move(provider))
            {
            }

            CameraMetaData(const CameraMetaData& other)
                :ip(other.ip), provider(other.provider)
            {
            }

            CameraMetaData& operator=(const CameraMetaData& other)
            {
                ip = other.ip;
                provider = other.provider;
                return *this;
            }

            CameraMetaData(CameraMetaData&& other) noexcept
                :ip(std::move(other.ip)), provider(std::move(other.provider))
            {
            }

            CameraMetaData& operator=(CameraMetaData&& other) noexcept
            {
                ip = std::move(other.ip);
                provider = std::move(other.provider);
                return *this;
            }

            ~CameraMetaData() = default;

        };


        enum class CameraObjectTrigger
        {
            Hardware,
            Software,
            Undefined
        };


        QVector<CameraMetaData> CheckCameraList();

    } // namespace rqw

} // namespace rw
