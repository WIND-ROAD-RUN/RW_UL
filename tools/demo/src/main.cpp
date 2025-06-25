#include <QPainter>
#include <QtWidgets/QApplication>
#include"rqw_DlgVersion.h"

#include"rqw_ImagePainter.h"
#include "rqw_ImageSaveEngine.h"

QImage createChapterImage(int chapterNumber) {
    QImage chapterImage(200, 200, QImage::Format_RGB32);
    chapterImage.fill(Qt::white); // 填充背景为白色

    QPainter painter(&chapterImage);
    painter.setRenderHint(QPainter::Antialiasing);

    // 设置字体
    QFont font("Arial", 40, QFont::Bold);
    painter.setFont(font);

    // 设置文本颜色
    painter.setPen(Qt::black);

    // 绘制章节数字
    QString text = QString("第 %1 章").arg(chapterNumber);
    QRect rect(0, 0, 200, 200); // 图像的绘制区域
    painter.drawText(rect, Qt::AlignCenter, text);

    painter.end();
    return chapterImage;
}

int main(int argc, char* argv[])
{
	QApplication a(argc, argv);
	DlgVersion w;

	rw::rqw::ImageSaveEngine imageSaveEngine;
	imageSaveEngine.setRootPath(R"(C:\Users\rw\Desktop\1)");
	imageSaveEngine.startEngine();
    imageSaveEngine.setSavePolicy(rw::rqw::ImageSaveEnginePolicy::MaxSaveImageNum);
    imageSaveEngine.setMaxSaveImageNum(20);
    imageSaveEngine.setSaveImgQuality(10);
    for (int i = 0;i<1000;i++)
    {
        imageSaveEngine.pushImage(createChapterImage(i));
    }

	w.loadVersionPath(R"(D:\zfkjData\SmartCroppingOfBags\Version\Version.txt)");
	w.show();

	return a.exec();
}
