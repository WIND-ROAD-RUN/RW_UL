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

		bool ZMotion::setIOOut(int portNum, bool state)
		{
			if (!_zMotion)
			{
				return false;
			}

			_zMotion->SetIOOut(portNum, state);
			return true;
		}

		bool ZMotion::SetIOOut(int axis, int ioNUm, bool state, int iotime)
		{
			if (!_zMotion)
			{
				return false;
			}

			_zMotion->SetIOOut(axis, ioNUm, state, iotime);
			return true;
		}

		bool ZMotion::setAxisPulse(int axis, float value)
		{
			if (!_zMotion)
			{
				return false;
			}

			_zMotion->SetAxisPulse(axis, value);
			return true;
		}

		bool ZMotion::setAxisRunSpeed(int axis, float value)
		{
			if (!_zMotion)
			{
				return false;
			}

			_zMotion->SetAxisRunSpeed(axis, value);
			return true;
		}
	}
}
