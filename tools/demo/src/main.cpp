#include <iostream>
#include <QPainter>
#include <random>
#include <QtWidgets/QApplication>
#include"PicturesPainter.h"
#include"LicenseValidation.h"
#include"rqwm_ModbusDeviceThread.hpp"

void performRandomIO(rw::rqwm::ModbusDeviceThreadSafe& modbusDeviceThread, int threadId) {
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_int_distribution<> dist(0, 4);

	for (int i = 0; i < 3000; ++i) { // 每个线程执行 30 次操作
		int operation = dist(gen);
		std::cout << modbusDeviceThread.getOState(rw::rqwm::ModbusO::Y03) << std::endl;
		std::this_thread::sleep_for(std::chrono::milliseconds(1)); // 模拟 I/O 操作的延迟
	}
}

int main(int argc, char* argv[])
{
	QApplication a(argc, argv);

	rw::rqwm::ModbusConfig config;
	config.ip = "192.168.1.199";
	config.port = 502;
	rw::rqwm::ModbusDeviceThreadSafe modbusDeviceThread(rw::rqwm::ModbusType::keRuiE, config);

	modbusDeviceThread.connect();
	modbusDeviceThread.setOState(rw::rqwm::ModbusO::Y03, true);
	// 创建 10 个线程
	std::vector<std::thread> threads;
	for (int i = 0; i < 10; ++i) {
		threads.emplace_back(performRandomIO, std::ref(modbusDeviceThread), i);
	}

	// 等待所有线程完成
	for (auto& thread : threads) {
		thread.join();
	}

	return a.exec();
}

//void DLgStudy::btn_paint_clicked()
//{
//    ui->btn_paint->setEnabled(false);
//    //openWindow();
//    HalconCpp::HObject rectangle;
//
//    bool iscreate = PaintRegion(rectangle);
//    if (iscreate)
//    {
//        LearingRegions.append(rectangle);
//        // 在listWidget中添加下标
//        int idx = LearingRegions.size() - 1;
//        ui->listWidget->addItem(QString::number(idx));
//    }
//    ui->btn_paint->setEnabled(true);
//
//}
//
//
////dispRegion();
//void DLgStudy::btn_study_clicked()
//{
//    isopen = false;
//    //彩色图片转化为灰度图像
//    HalconCpp::HObject grayimage;
//    HalconCpp::HObject colorimage;
//    colorimage = modelImage.Clone();
//    HalconCpp::Rgb1ToGray(colorimage, &grayimage);
//
//    // 合并所有区域
//    HalconCpp::HObject region;
//    if (!LearingRegions.isEmpty()) {
//        region = LearingRegions[0];
//        for (int i = 1; i < LearingRegions.size(); ++i) {
//            HalconCpp::HObject temp;
//            HalconCpp::Union2(region, LearingRegions[i], &temp);
//            region = temp;
//        }
//    }
//
//    // 灰度图像 grayimage 已经存在
//    HalconCpp::HObject reduceimage;
//    if (region.IsInitialized()) {
//        HalconCpp::ReduceDomain(grayimage, region, &reduceimage);
//    }
//
//
//
//
//
//    // 从界面获取阈值参数
//    double a = ui->btn_selectPixMin->text().toDouble();
//
//    // 边缘提取
//    HalconCpp::HObject ho_Border;
//    HalconCpp::EdgesSubPix(reduceimage, &ho_Border, "canny", 3, 5, 10);
//
//    // 选择轮廓长度大于a的边缘
//    HalconCpp::HObject ho_SelectedBorder;
//    HalconCpp::SelectShapeXld(ho_Border, &ho_SelectedBorder, "contlength", "and", a, 10000000);
//
//    // 统计轮廓数量
//    HalconCpp::HTuple number;
//    HalconCpp::CountObj(ho_SelectedBorder, &number);
//
//    if (number.D() > 0)
//    {
//        // 创建形状模板
//        HalconCpp::HTuple hv_ModelID;
//        HalconCpp::CreateShapeModelXld(
//            ho_SelectedBorder,
//            "auto",
//            HalconCpp::HTuple(0).TupleRad(),
//            HalconCpp::HTuple(360).TupleRad(),
//            "auto",
//            "auto",
//            "use_polarity",
//            5,
//            &hv_ModelID
//        );
//
//
//        // 获取模板的轮廓
//        HalconCpp::HObject modelContours;
//        HalconCpp::GetShapeModelContours(&modelContours, hv_ModelID, 1);
//        // 查找模板
//        HalconCpp::HTuple hv_Rowmodel, hv_Columnmodel, hv_Anglemodel, hv_Score;
//        HalconCpp::FindShapeModel(
//            reduceimage,
//            hv_ModelID,
//            0,
//            HalconCpp::HTuple(120).TupleRad(),
//            0.2,
//            1,
//            0.5,
//            "least_squares",
//            0,
//            0.9,
//            &hv_Rowmodel,
//            &hv_Columnmodel,
//            &hv_Anglemodel,
//            &hv_Score
//        );
//
//        if (hv_Rowmodel.TupleLength() > 0)
//        {
//            // 取第一个匹配结果
//            double row = hv_Rowmodel[0].D();
//            double col = hv_Columnmodel[0].D();
//            double angle = hv_Anglemodel[0].D();
//
//            // 计算仿射变换矩阵
//            HalconCpp::HTuple hv_HomMat2D;
//            HalconCpp::VectorAngleToRigid(
//                0, 0, 0, // 模板原点和角度
//                row, col, angle, // 匹配到的位置和角度
//                &hv_HomMat2D
//            );
//
//            // 变换模板轮廓
//            HalconCpp::HObject modelContoursTrans;
//            HalconCpp::AffineTransContourXld(
//                modelContours,
//                &modelContoursTrans,
//                hv_HomMat2D
//            );
//
//            // 显示匹配结果
//            HalconCpp::SetColor(windowhandle, "blue");
//            HalconCpp::DispObj(colorimage, windowhandle);
//
//            HalconCpp::DispObj(modelContoursTrans, windowhandle);
//        }
//        else
//        {
//
//            //匹配失败
//        }
//    }
//    else
//    {
//
//        //没有找到轮廓
//
//    }
//
//
//
//
//}