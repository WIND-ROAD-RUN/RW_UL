#include <iostream>
#include <QPainter>
#include <random>
#include <QtWidgets/QApplication>
#include"PicturesPainterVersionDunDai.h"

// 生成随机颜色（避免过暗）
static QColor randomColor(std::mt19937& rng)
{
    std::uniform_int_distribution<int> d(60, 220); // 适中亮度
    return QColor(d(rng), d(rng), d(rng));
}

// 随机生成若干 RectangeConfig，只用于填充 listView
static std::vector<rw::rqw::RectangeConfig> buildRandomConfigs(
    int count,
    unsigned seed = std::random_device{}())
{
    std::mt19937 rng(seed);
    std::vector<rw::rqw::RectangeConfig> cfgs;
    cfgs.reserve(count);

    for (int i = 0; i < count; ++i)
    {
        rw::rqw::RectangeConfig c;
        c.classid = i;                              // 顺序 id
        c.color = randomColor(rng);                 // 随机颜色
        c.name = QString("Class123123_%1").arg(i);        // 名称
        c.descrption = QString("随机测试类别 %1").arg(i);
        cfgs.push_back(c);
    }
    return cfgs;
}

int main(int argc, char* argv[])
{
	QApplication a(argc, argv);

	PicturesPainterVersionDunDai painter;
	QImage img;
	img.load(R"(C:\Users\zfkj4090\Desktop\temp\total.png)");
	painter.setImage(img);
	// 仅生成 listView 选项（类别配置），不预先添加任何绘制框z
	int configCount = 8; // 可根据需要调整或改为读取 argv
	auto configs = buildRandomConfigs(configCount);
	painter.setRectangleConfigs(configs);

	painter.show();

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