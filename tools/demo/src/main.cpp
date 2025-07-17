#include <QPainter>
#include <QtWidgets/QApplication>
#include"PicturesPainter.h"

int main(int argc, char* argv[])
{
	QApplication a(argc, argv);
	// 1. 创建一个配置 vector
	std::vector<RectangeConfig> configs;

	// 可以根据需要往 configs 填入内容，例如
	configs.push_back(RectangeConfig{ 1, QColor(Qt::red), "黑疤", "黑疤" });
	configs.push_back(RectangeConfig{ 2, QColor(Qt::green), "疏档", "疏档" });
    configs.push_back(RectangeConfig{ 2, QColor(Qt::blue), "框", "1" });

	PicturesPainter p;
	p.setRectangleConfigs(configs);

	// 2. 读取图片
	QString imagePath = R"(C:\Users\zzw\Desktop\123.jpeg)"; // 绝对路径或相对路径
	QImage image(imagePath);
	if (image.isNull()) {
		qDebug() << "加载图片失败:" << imagePath;
		return -1;
	}

	p.setImage(image); 
    //p.setAspectRatio(300, 200);

    std::vector<PicturesPainter::PainterRectangleInfo> rectangles;

    // 框1
    rectangles.push_back({
        {100, 100},   // leftTop
        {200, 100},   // rightTop
        {100, 200},   // leftBottom
        {200, 200},   // rightBottom
        150,          // center_x
        150,          // center_y
        100,          // width
        100,          // height
        10000,        // area
        1,            // classId
        0.92          // score
        });

    // 框2
    rectangles.push_back({
        {50, 50},
        {120, 50},
        {50, 120},
        {120, 120},
        85,    // center_x
        85,    // center_y
        70,    // width
        70,    // height
        4900,  // area
        2,
        0.87
        });

    // 框3
    rectangles.push_back({
        {220, 80},
        {320, 80},
        {220, 180},
        {320, 180},
        270,    // center_x
        130,    // center_y
        100,    // width
        100,    // height
        10000,  // area
        1,
        0.75
        });

    p.setDrawnRectangles(rectangles);
	auto result=p.exec();
    if (result==QDialog::Accepted)
    {

       auto rets= p.getRectangleConfigs();
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
