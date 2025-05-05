#include "scc_Motion.h"


namespace zwy {
	namespace scc {
		Motion::Motion()
		{
		}
		bool Motion::OpenBoard(char* ipAdress)
		{
			// qDebug()<<"运行";
			char* ip_addr = (char*)ipAdress;
			int ret = ZAux_OpenEth(ip_addr, &g_handle);
			//qDebug()<<ipAdress;
			if (ERR_SUCCESS != ret)
			{
				g_handle = nullptr;
				//qDebug()<<"fale";
				isOPen = false;
				return false;
			}
			else
			{
				// qDebug()<<"success";
				isOPen = true;
				ZMC_SetTimeOut(g_handle, 100);

				return true;
			}
		}
		void Motion::CloseBoared()
		{
			if (g_handle != nullptr)
			{
				ZAux_Close(g_handle);
				//关闭连接

				g_handle = nullptr;
			}
		}
		//bool Motion::GetBoardStatue()
		//{
		//}
		void Motion::AxisRun(int axis, float value)
		{
			ZAux_Direct_Single_Vmove(g_handle, axis, -value);
		}

		//设置轴的类型
		//1为脉冲方向方式的步进或伺服
		//2为模拟型号控制伺服
		void Motion::SetAxisType(int axis, int value)
		{
			if (g_handle != nullptr)
			{
				int ret = ZAux_Direct_SetAtype(g_handle, axis, value);
			}
		}
		//设置脉冲当量
		void Motion::SetAxisPulse(int axis, float value)
		{
			if (g_handle != nullptr)
			{
				int  ret = ZAux_Direct_SetUnits(g_handle, axis, value);
			}
		}
		//设置轴的运动速度
		void Motion::SetAxisRunSpeed(int axis, float value)
		{
			if (g_handle != nullptr)
			{
				ZAux_Direct_SetSpeed(g_handle, axis, value);
			}
		}

		//获取当前轴的位置
		void Motion::GetAxisLocation(int axis, float& value)
		{
			if (g_handle != nullptr)
			{
				ZAux_Direct_GetDpos(g_handle, axis, &value);
			}
		}

		void  Motion::SetModbus(uint16 adress, uint16 num, uint8 value)
		{
			uint8 v;
			v = value;
			int ret = ZAux_Modbus_Set0x(g_handle, adress, num, &v);
		}

		void Motion::SetIOOut(int axis, int ioNUm, bool state, int iotime)
		{
			if (g_handle != nullptr)
			{
				int s = -1;
				if (state == true)
				{
					s = 1;
					ZAux_Direct_MoveOp2(g_handle, axis, ioNUm, s, iotime);
				}
				else
				{
					s = 0;
					ZAux_Direct_MoveOp2(g_handle, axis, ioNUm, s, iotime);
				}
			}
		}

		//获取输入IO状态
		bool  Motion::GetIOIn(int portNum)
		{
			if (g_handle != nullptr)
			{
				unsigned int in_value;
				int ret = ZAux_Direct_GetIn(g_handle, portNum, &in_value);
				if (in_value == 0)
				{
					return false;
				}
				else
				{
					return true;
				}
			}
		}
		//获取输出IO状态
		bool  Motion::GetIOOut(int portNum)
		{
			if (g_handle != nullptr)
			{
				unsigned int out_value;
				int  ret = ZAux_Direct_GetOp(g_handle, portNum, &out_value);
				if (out_value == 0)
				{
					return false;
				}
				else
				{
					return true;
				}
			}
		}
		//设置输出IO状态
		void Motion::SetIOOut(int portNum, bool state)
		{
			if (g_handle != nullptr)
			{
				int s = -1;
				if (state == true)
				{
					s = 1;
					int ret = ZAux_Direct_SetOp(g_handle, portNum, s);
				}
				else
				{
					s = 0;
					int ret = ZAux_Direct_SetOp(g_handle, portNum, s);
				}
			}
		}

		//单轴停止运动
		void Motion::Single_Stop(int axis)
		{
			if (g_handle != nullptr)
			{
			}
		}
		//所有轴停止运动
		void Motion::StopAllAxis()
		{
			if (g_handle != nullptr)
			{
				ZAux_Direct_Single_Cancel(g_handle, 0, 2);
			}
		}
		//单轴运动
		void Motion::Single_Move(int axis, double dir)
		{
			if (g_handle)
			{
				int     ret = ZAux_Direct_Single_Move(g_handle, axis, dir);
			}
		}
		//单轴运动
		void  Motion::Single_Move(int axis, int dir, float speed, float acc, float dec, float Units)
		{
			if (g_handle)
			{
				if (dir > 0)
				{
					dir = 1;
				}
				else
				{
					dir = -1;
				}
				int ret = ZAux_Direct_SetAtype(g_handle, axis, 2);//设置为脉冲式伺服
				ret = ZAux_Direct_SetSpeed(g_handle, axis, speed); //设置轴 0 速度为 200units/s
				ret = ZAux_Direct_SetUnits(g_handle, axis, Units);//设置轴 0 脉冲当量为 200units/s
				ret = ZAux_Direct_SetAccel(g_handle, axis, acc); //设置轴 0 加速度为 2000units/s/s
				ret = ZAux_Direct_SetDecel(g_handle, axis, dec); //设置轴 0 减速度为 2000units/s/s
				ret = ZAux_Direct_SetSramp(g_handle, axis, 100); //设置轴 0 S曲线时间 100(梯形加减速)
				ret = ZAux_Direct_Single_Vmove(g_handle, axis, dir);
			}
		}

		//轴位置清零
		void  Motion::SetLocationZero(int axis)
		{
			if (g_handle)
			{
				int ret = ZAux_Direct_SetDpos(g_handle, axis, 0);
			}
		}

		bool Motion::getBoardState()
		{
			uint8 state = -1;
			if (g_handle)
			{
				ZMC_GetState(g_handle, &state);
			}
			if (state == 1)
			{
				return true;
			}
			else
			{
				return false;
			}
		}

		float Motion::getAxisSpeed(int axis)
		{
			float speed = 0;
			ZAux_Direct_GetVpSpeed(g_handle, axis, &speed);
			return speed;
		}

		//  设置轴运动加速度
		void Motion::SetAxisAcc(int axis, float value)
		{
			ZAux_Direct_SetAccel(g_handle, axis, value);
		}
		//  设置轴运动减速度
		void Motion::SetAxisDec(int axis, float value)
		{
			ZAux_Direct_SetDecel(g_handle, axis, value);
		}

		bool Motion::GetAllIOIN(int portNum)
		{
			ZAux_Direct_GetInMulti(g_handle, 8, 11, &portNum);
			return 1;
		}

		void  Motion::SetModbus(uint16 adress, uint16 num, float value)
		{
			float v;
			v = value;
			// int ret = ZAux_Modbus_Set0x(g_handle, adress, num,  &v);
			int ret = ZAux_Modbus_Set4x_Float(g_handle, adress, num, &v);
		}
		void  Motion::GetModbus(uint16 adress, uint16 num, float& value)
		{
			float v;
			v = value;
			// int ret = ZAux_Modbus_Set0x(g_handle, adress, num,  &v);
			int ret = ZAux_Modbus_Get4x_Float(g_handle, adress, num, &v);
			value = v;
		}
	}
}