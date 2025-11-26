#include"actCry_HWID.hpp"

#include "cla_ActivationCodeGenerator.hpp"
#include "hoei_HardwareInfo.hpp"

#include <windows.h>


namespace rw
{
	namespace actCry
	{
		std::string HWID::generate(const HWIDGenerateCfg& cfg)
		{
			auto hardwareInfo = hoei::HardwareInfoFactory::createHardwareInfo();
			auto HWID = hardwareInfo.motherBoard.UUID;

			cla::ActivationConfig config;
			config.insert("ProductName", cfg.productName);
			config.insert(HWID, HWID);

			rw::cla::ActivationCodeGenerator generator(config);
			auto code = generator.generateActivationCode(HWID);

			return code.str;
		}

		static std::wstring utf8_to_wstring(const std::string& s)
		{
			if (s.empty()) return {};
			int len = ::MultiByteToWideChar(CP_UTF8, 0, s.data(), static_cast<int>(s.size()), nullptr, 0);
			if (len <= 0) return {};
			std::wstring w;
			w.resize(len);
			::MultiByteToWideChar(CP_UTF8, 0, s.data(), static_cast<int>(s.size()), &w[0], len);
			return w;
		}

		static std::string wstring_to_utf8(const std::wstring& ws)
		{
			if (ws.empty()) return {};
			int len = ::WideCharToMultiByte(CP_UTF8, 0, ws.data(), static_cast<int>(ws.size()), nullptr, 0, nullptr, nullptr);
			if (len <= 0) return {};
			std::string s;
			s.resize(len);
			::WideCharToMultiByte(CP_UTF8, 0, ws.data(), static_cast<int>(ws.size()), &s[0], len, nullptr, nullptr);
			return s;
		}

		bool HWID::save(const std::string& hwid, const HWIDSaveRegistryCfg& cfg)
		{
			// 如果没有提供 name，则返回失败
			if (cfg.name.empty()) return false;

			// 存储位置：HKEY_CURRENT_USER\Software\RW\ActCry，值名 使用 cfg.name
			// 使用 HKCU 避免需要管理员权限。若需要机器范围存储可改为 HKEY_LOCAL_MACHINE（需管理员）。
			const std::wstring keyPath = utf8_to_wstring(cfg.keyPath);
			HKEY hKey = nullptr;
			LONG rc = RegCreateKeyExW(
				HKEY_CURRENT_USER,
				keyPath.c_str(),
				0,
				nullptr,
				REG_OPTION_NON_VOLATILE,
				KEY_WRITE,
				nullptr,
				&hKey,
				nullptr);
			if (rc != ERROR_SUCCESS || hKey == nullptr)
			{
				return false;
			}

			std::wstring valueName = utf8_to_wstring(cfg.name);
			std::wstring wvalue = utf8_to_wstring(hwid);
			DWORD dataSize = static_cast<DWORD>((wvalue.size() + 1) * sizeof(wchar_t)); // 包含终止 null

			rc = RegSetValueExW(
				hKey,
				valueName.c_str(),
				0,
				REG_SZ,
				reinterpret_cast<const BYTE*>(wvalue.c_str()),
				dataSize);

			RegCloseKey(hKey);
			return rc == ERROR_SUCCESS;
		}

		std::string HWID::load(const HWIDSaveRegistryCfg& cfg)
		{
			// 参数检查
			if (cfg.name.empty()) return {};

			const std::wstring keyPath = utf8_to_wstring(cfg.keyPath);
			HKEY hKey = nullptr;
			LONG rc = RegOpenKeyExW(HKEY_CURRENT_USER, keyPath.c_str(), 0, KEY_READ, &hKey);
			if (rc != ERROR_SUCCESS || hKey == nullptr)
			{
				return {};
			}

			std::wstring valueName = utf8_to_wstring(cfg.name);

			// 首先查询所需缓冲区大小
			DWORD type = 0;
			DWORD dataSize = 0;
			rc = RegQueryValueExW(hKey, valueName.c_str(), NULL, &type, NULL, &dataSize);
			if (rc != ERROR_SUCCESS || (type != REG_SZ && type != REG_EXPAND_SZ))
			{
				RegCloseKey(hKey);
				return {};
			}

			if (dataSize == 0)
			{
				RegCloseKey(hKey);
				return {};
			}

			// dataSize 以字节为单位，分配 wchar_t 缓冲区
			std::vector<wchar_t> buffer((dataSize / sizeof(wchar_t)) + 1);
			rc = RegQueryValueExW(hKey, valueName.c_str(), NULL, NULL, reinterpret_cast<LPBYTE>(buffer.data()), &dataSize);
			if (rc != ERROR_SUCCESS)
			{
				RegCloseKey(hKey);
				return {};
			}

			// 确保以 null 结尾
			buffer.back() = L'\0';
			std::wstring wvalue(buffer.data());

			RegCloseKey(hKey);

			return wstring_to_utf8(wvalue);
		}
	}
}
