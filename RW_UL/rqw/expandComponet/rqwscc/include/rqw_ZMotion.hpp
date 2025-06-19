#pragma once
#include <memory>
#include <QString>

namespace zwy
{
	namespace scc
	{
		class Motion;
	}
}

namespace rw
{
	namespace rqw
	{
		class ZMotion
		{
		private:
			std::unique_ptr<zwy::scc::Motion> _zMotion{nullptr};
		public:
			explicit ZMotion(const QString &ip);
			ZMotion();
		private:
			QString _ip{};
		public:
			void setIp(const QString & ip);
			QString getIp(const QString& ip);
		public:
			bool connect();

			bool getConnectState(bool &isGet);
			bool getConnectState();

			bool disConnect(bool& isGet);
			bool disConnect();
		public:
			bool getIOIn(int portNum);
			bool getIOIn(int portNum, bool& isGet);

			bool getIOOut(int portNum);
			bool getIOOut(int portNum, bool& isGet);

			bool setIOOut(int portNum, bool state);
			bool SetIOOut(int axis, int ioNUm, bool state, int iotime);
		public:
			bool setAxisPulse(int axis, float value);
			bool setAxisRunSpeed(int axis, float value);
		};

	}	
}
