#include"ButtonUtilty.h"

QString VersionInfo::Version = QStringLiteral("内部测试版：0.0.0.1.20250625");

ControlLine ControlLines::blowLine1{ 5,5 };
ControlLine ControlLines::blowLine2{ 1,4 };
ControlLine ControlLines::blowLine3{ 2,3 };
ControlLine ControlLines::blowLine4{ 3,2 };

int ControlLines::stopIn=0;
int ControlLines::startIn=0;
int ControlLines::airWarnIn=0;
int ControlLines::shutdownComputerIn=0;
int ControlLines::camer1In=0;
int ControlLines::camer2In=0;
int ControlLines::camer3In=0;
int ControlLines::camer4In=0;

int ControlLines::motoPowerOut=0;
int ControlLines::beltAsis=0;
int ControlLines::warnGreenOut=0;
int ControlLines::warnRedOut=0;
int ControlLines::upLightOut=0;
int ControlLines::sideLightOut=0;
int ControlLines::downLightOut=0;
int ControlLines::strobeLightOut=0;