#include "halcon_pch_t.hpp"
#include"halconcpp/HalconCpp.h"

using namespace HalconCpp;

TEST(HalconExample, DetectCircles)
{
    try
    {
        // 初始化 Halcon 窗口
        HObject ho_Image, ho_Regions, ho_Circles;
        HTuple hv_WindowID;

        // 加载图像
        ReadImage(&ho_Image, "fabrik");

        // 创建窗口以显示图像
        SetWindowAttr("background_color", "black");
        OpenWindow(0, 0, 512, 512, 0, "visible", "", &hv_WindowID);
        HDevWindowStack::Push(hv_WindowID);

        // 显示图像
        if (HDevWindowStack::IsOpen())
            DispObj(ho_Image, HDevWindowStack::GetActive());

        // 检测圆形对象
        Threshold(ho_Image, &ho_Regions, 128, 255);
        Connection(ho_Regions, &ho_Regions);
        SelectShape(ho_Regions, &ho_Circles, "circularity", "and", 0.8, 1.0);

        // 显示检测到的圆
        if (HDevWindowStack::IsOpen())
            DispObj(ho_Circles, HDevWindowStack::GetActive());

        // 关闭窗口
        CloseWindow(hv_WindowID);

        SUCCEED();
    }
    catch (HException& e)
    {
        std::cerr << "Halcon Error: " << e.ErrorMessage() << std::endl;
        FAIL();
    }
}