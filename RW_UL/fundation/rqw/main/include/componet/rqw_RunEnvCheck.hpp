#pragma once

#include<QString>

namespace rw {
	namespace rqw {
		class RunEnvCheck {
		public:
			static bool isSingleInstance(const QString& instanceName);
			static bool isProcessRunning(const QString& processName);
			static bool isFileExist(const QString& filePath);
		};
	}
}