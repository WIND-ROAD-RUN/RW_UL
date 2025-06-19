#include"rqw_ZMotion.hpp"

#include"scc_motion.h"

namespace rw
{
	namespace rqw
	{
		ZMotion::ZMotion(const QString& ip)
			:_ip(ip)
		{
			_zMotion = std::make_unique<zwy::scc::Motion>();
			_zMotion->OpenBoard(ip.toStdString());
		}

		ZMotion::ZMotion()
		{
		}

		void ZMotion::setIp(const QString& ip)
		{
			_ip = ip;
		}

		QString ZMotion::getIp(const QString& ip)
		{
			return _ip;
		}

		bool ZMotion::connect()
		{
			_zMotion = std::make_unique<zwy::scc::Motion>();
			return _zMotion->OpenBoard(_ip.toStdString());
		}

		bool ZMotion::getConnectState(bool& isGet)
		{
			if (!_zMotion)
			{
				isGet = false;
				return false;
			}
			isGet = true;
			return _zMotion->getBoardState();
		}

		bool ZMotion::getConnectState()
		{
			bool temp{true};
			return getConnectState(temp);
		}

		bool ZMotion::disConnect(bool& isGet)
		{
			if (!_zMotion)
			{
				isGet = false;
				return false;
			}
			isGet = true;
			auto result= _zMotion->CloseBoared();
			_zMotion.reset();
			return result;
		}

		bool ZMotion::disConnect()
		{
			bool temp{ true };
			return disConnect(temp);
		}

		bool ZMotion::getIOIn(int portNum)
		{
			bool temp{true};
			return getIOIn(portNum, temp);
		}

		bool ZMotion::getIOIn(int portNum, bool& isGet)
		{
			if (!_zMotion)
			{
				isGet = false;
				return false;
			}
			isGet = true;
			return _zMotion->GetIOIn(portNum);
		}


		bool ZMotion::getIOOut(int portNum)
		{
			bool temp{ true };
			return getIOOut(portNum, temp);
		}

		bool ZMotion::getIOOut(int portNum, bool& isGet)
		{
			if (!_zMotion)
			{
				isGet = false;
				return false;
			}
			isGet = true;
			return _zMotion->GetIOOut(portNum);
		}
	}
}
