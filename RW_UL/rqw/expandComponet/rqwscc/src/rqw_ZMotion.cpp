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

			return _zMotion->SetIOOut(portNum, state);
		}

		bool ZMotion::SetIOOut(int axis, int ioNUm, bool state, int iotime)
		{
			if (!_zMotion)
			{
				return false;
			}

			
			return _zMotion->SetIOOut(axis, ioNUm, state, iotime);
		}

		bool ZMotion::setAxisPulse(int axis, float value)
		{
			if (!_zMotion)
			{
				return false;
			}

			return _zMotion->SetAxisPulse(axis, value);
		}

		bool ZMotion::setAxisRunSpeed(int axis, float value)
		{
			if (!_zMotion)
			{
				return false;
			}

			return _zMotion->SetAxisRunSpeed(axis, value);
		}

		bool ZMotion::setAxisAcc(int axis, float value)
		{
			if (!_zMotion)
			{
				return false;
			}

			return _zMotion->SetAxisAcc(axis, value);
		}

		bool ZMotion::setAxisDec(int axis, float value)
		{
			if (!_zMotion)
			{
				return false;
			}

			return _zMotion->SetAxisDec(axis, value);
		}

		bool ZMotion::setAxisRun(int axis, float value)
		{
			if (!_zMotion)
			{
				return false;
			}

			return _zMotion->SetAxisRun(axis, value);
		}

		bool ZMotion::stopAllAxis()
		{
			if (!_zMotion)
			{
				return false;
			}

			return _zMotion->StopAllAxis();
		}

		float ZMotion::getAxisLocation(int axis, bool& isGet)
		{
			float result{0};
			if (!_zMotion)
			{
				isGet = false;
				return result;
			}
			isGet = _zMotion->GetAxisLocation(axis, result);
			return result;
		}

		float ZMotion::getAxisLocation(int axis)
		{
			bool isGet{false};
			return getAxisLocation(axis,isGet);
		}

		bool ZMotion::singleStop(int axis)
		{
			if (!_zMotion)
			{
				return false;
			}

			return _zMotion->Single_Stop(axis);
		}
	}
}
