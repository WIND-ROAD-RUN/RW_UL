#include "DlgProductScore.h"

#include <QMessageBox>

#include "GlobalStruct.hpp"
#include "NumberKeyboard.h"

DlgProductScoreSmartCroppingOfBags::DlgProductScoreSmartCroppingOfBags(QWidget *parent)
	: QDialog(parent)
	, ui(new Ui::DlgProductScoreClass())
{
	ui->setupUi(this);

	build_ui();

	build_connect();
}

DlgProductScoreSmartCroppingOfBags::~DlgProductScoreSmartCroppingOfBags()
{
	delete ui;
}

void DlgProductScoreSmartCroppingOfBags::build_ui()
{
	read_config();
}

void DlgProductScoreSmartCroppingOfBags::read_config()
{
	auto& ScoreConfig = GlobalStructDataSmartCroppingOfBags::getInstance().scoreConfig;

    ui->ckb_heiba->setChecked(ScoreConfig.heiba);
    ui->btn_heibascore->setText(QString::number(ScoreConfig.heibascore));
    ui->btn_heibaarea->setText(QString::number(ScoreConfig.heibaarea));

    ui->ckb_shudang->setChecked(ScoreConfig.shudang);
    ui->btn_shudangscore->setText(QString::number(ScoreConfig.shudangscore));
    ui->btn_shudangarea->setText(QString::number(ScoreConfig.shudangarea));

    ui->ckb_huapo->setChecked(ScoreConfig.huapo);
    ui->btn_huaposcore->setText(QString::number(ScoreConfig.huaposcore));
    ui->btn_huapoarea->setText(QString::number(ScoreConfig.huapoarea));

    ui->ckb_jietou->setChecked(ScoreConfig.jietou);
    ui->btn_jietouscore->setText(QString::number(ScoreConfig.jietouscore));
    ui->btn_jietouarea->setText(QString::number(ScoreConfig.jietouarea));

    ui->ckb_guasi->setChecked(ScoreConfig.guasi);
    ui->btn_guasiscore->setText(QString::number(ScoreConfig.guasiscore));
    ui->btn_guasiarea->setText(QString::number(ScoreConfig.guasiarea));

    ui->ckb_podong->setChecked(ScoreConfig.podong);
    ui->btn_podongscore->setText(QString::number(ScoreConfig.podongscore));
    ui->btn_podongarea->setText(QString::number(ScoreConfig.podongarea));

    ui->ckb_zangwu->setChecked(ScoreConfig.zangwu);
    ui->btn_zangwuscore->setText(QString::number(ScoreConfig.zangwuscore));
    ui->btn_zangwuarea->setText(QString::number(ScoreConfig.zangwuarea));

    ui->ckb_noshudang->setChecked(ScoreConfig.noshudang);
    ui->btn_noshudangscore->setText(QString::number(ScoreConfig.noshudangscore));
    ui->btn_noshudangarea->setText(QString::number(ScoreConfig.noshudangarea));

    ui->ckb_modian->setChecked(ScoreConfig.modian);
    ui->btn_modianscore->setText(QString::number(ScoreConfig.modianscore));
    ui->btn_modianarea->setText(QString::number(ScoreConfig.modianarea));

    ui->ckb_loumo->setChecked(ScoreConfig.loumo);
    ui->btn_loumoscore->setText(QString::number(ScoreConfig.loumoscore));
    ui->btn_loumoarea->setText(QString::number(ScoreConfig.loumoarea));

    ui->ckb_xishudang->setChecked(ScoreConfig.xishudang);
    ui->btn_xishudangscore->setText(QString::number(ScoreConfig.xishudangscore));
    ui->btn_xishudangarea->setText(QString::number(ScoreConfig.xishudangarea));

    ui->ckb_erweima->setChecked(ScoreConfig.erweima);
    ui->btn_erweimascore->setText(QString::number(ScoreConfig.erweimascore));
    ui->btn_erweimaarea->setText(QString::number(ScoreConfig.erweimaarea));

    ui->ckb_damodian->setChecked(ScoreConfig.damodian);
    ui->btn_damodianscore->setText(QString::number(ScoreConfig.damodianscore));
    ui->btn_damodianarea->setText(QString::number(ScoreConfig.damodianarea));

    ui->ckb_kongdong->setChecked(ScoreConfig.kongdong);
    ui->btn_kongdongscore->setText(QString::number(ScoreConfig.kongdongscore));
    ui->btn_kongdongarea->setText(QString::number(ScoreConfig.kongdongarea));

    ui->ckb_sebiao->setChecked(ScoreConfig.sebiao);
    ui->btn_sebiaoscore->setText(QString::number(ScoreConfig.sebiaoscore));
    ui->btn_sebiaoarea->setText(QString::number(ScoreConfig.sebiaoarea));

    ui->ckb_yinshuaquexian->setChecked(ScoreConfig.yinshuaquexian);
    ui->btn_yinshuaquexianscore->setText(QString::number(ScoreConfig.yinshuaquexianscore));
    ui->btn_yinshuaquexianarea->setText(QString::number(ScoreConfig.yinshuaquexianarea));

    ui->ckb_xiaopodong->setChecked(ScoreConfig.xiaopodong);
    ui->btn_xiaopodongscore->setText(QString::number(ScoreConfig.xiaopodongscore));
    ui->btn_xiaopodongarea->setText(QString::number(ScoreConfig.xiaopodongarea));

    ui->ckb_jiaodai->setChecked(ScoreConfig.jiaodai);
    ui->btn_jiaodaiscore->setText(QString::number(ScoreConfig.jiaodaiscore));
    ui->btn_jiaodaiarea->setText(QString::number(ScoreConfig.jiaodaiarea));

}

void DlgProductScoreSmartCroppingOfBags::build_connect()
{
    QObject::connect(ui->btn_close, &QPushButton::clicked,
        this, &DlgProductScoreSmartCroppingOfBags::btn_close_clicked);
    // 黑疤
    QObject::connect(ui->ckb_heiba, &QCheckBox::clicked,
        this, &DlgProductScoreSmartCroppingOfBags::ckb_heiba_checked);
    QObject::connect(ui->btn_heibascore, &QPushButton::clicked,
        this, &DlgProductScoreSmartCroppingOfBags::btn_heibascore_clicked);
    QObject::connect(ui->btn_heibaarea, &QPushButton::clicked, 
        this, &DlgProductScoreSmartCroppingOfBags::btn_heibaarea_clicked);

    // 疏档
    QObject::connect(ui->ckb_shudang, &QCheckBox::clicked,
        this, &DlgProductScoreSmartCroppingOfBags::ckb_shudang_checked);
    QObject::connect(ui->btn_shudangscore, &QPushButton::clicked,
        this, &DlgProductScoreSmartCroppingOfBags::btn_shudangscore_clicked);
    QObject::connect(ui->btn_shudangarea, &QPushButton::clicked,
        this, &DlgProductScoreSmartCroppingOfBags::btn_shudangarea_clicked);

    // 划破
    QObject::connect(ui->ckb_huapo, &QCheckBox::clicked,
        this, &DlgProductScoreSmartCroppingOfBags::ckb_huapo_checked);
    QObject::connect(ui->btn_huaposcore, &QPushButton::clicked, 
        this, &DlgProductScoreSmartCroppingOfBags::btn_huaposcore_clicked);
    QObject::connect(ui->btn_huapoarea, &QPushButton::clicked,
        this, &DlgProductScoreSmartCroppingOfBags::btn_huapoarea_clicked);

    // 接头
    QObject::connect(ui->ckb_jietou, &QCheckBox::clicked,
        this, &DlgProductScoreSmartCroppingOfBags::ckb_jietou_checked);
    QObject::connect(ui->btn_jietouscore, &QPushButton::clicked, 
        this, &DlgProductScoreSmartCroppingOfBags::btn_jietouscore_clicked);
    QObject::connect(ui->btn_jietouarea, &QPushButton::clicked,
        this, &DlgProductScoreSmartCroppingOfBags::btn_jietouarea_clicked);

    // 挂丝
    QObject::connect(ui->ckb_guasi, &QCheckBox::clicked, 
        this, &DlgProductScoreSmartCroppingOfBags::ckb_guasi_checked);
    QObject::connect(ui->btn_guasiscore, &QPushButton::clicked,
        this, &DlgProductScoreSmartCroppingOfBags::btn_guasiscore_clicked);
    QObject::connect(ui->btn_guasiarea, &QPushButton::clicked, 
        this, &DlgProductScoreSmartCroppingOfBags::btn_guasiarea_clicked);

    // 破洞
    QObject::connect(ui->ckb_podong, &QCheckBox::clicked,
        this, &DlgProductScoreSmartCroppingOfBags::ckb_podong_checked);
    QObject::connect(ui->btn_podongscore, &QPushButton::clicked,
        this, &DlgProductScoreSmartCroppingOfBags::btn_podongscore_clicked);
    QObject::connect(ui->btn_podongarea, &QPushButton::clicked,
        this, &DlgProductScoreSmartCroppingOfBags::btn_podongarea_clicked);

    // 脏污
    QObject::connect(ui->ckb_zangwu, &QCheckBox::clicked,
        this, &DlgProductScoreSmartCroppingOfBags::ckb_zangwu_checked);
    QObject::connect(ui->btn_zangwuscore, &QPushButton::clicked, 
        this, &DlgProductScoreSmartCroppingOfBags::btn_zangwuscore_clicked);
    QObject::connect(ui->btn_zangwuarea, &QPushButton::clicked,
        this, &DlgProductScoreSmartCroppingOfBags::btn_zangwuarea_clicked);

    // 无疏档
    QObject::connect(ui->ckb_noshudang, &QCheckBox::clicked, 
        this, &DlgProductScoreSmartCroppingOfBags::ckb_noshudang_checked);
    QObject::connect(ui->btn_noshudangscore, &QPushButton::clicked,
        this, &DlgProductScoreSmartCroppingOfBags::btn_noshudangscore_clicked);
    QObject::connect(ui->btn_noshudangarea, &QPushButton::clicked,
        this, &DlgProductScoreSmartCroppingOfBags::btn_noshudangarea_clicked);

    // 墨点
    QObject::connect(ui->ckb_modian, &QCheckBox::clicked, 
        this, &DlgProductScoreSmartCroppingOfBags::ckb_modian_checked);
    QObject::connect(ui->btn_modianscore, &QPushButton::clicked, 
        this, &DlgProductScoreSmartCroppingOfBags::btn_modianscore_clicked);
    QObject::connect(ui->btn_modianarea, &QPushButton::clicked,
        this, &DlgProductScoreSmartCroppingOfBags::btn_modianarea_clicked);

    // 漏膜
    QObject::connect(ui->ckb_loumo, &QCheckBox::clicked,
        this, &DlgProductScoreSmartCroppingOfBags::ckb_loumo_checked);
    QObject::connect(ui->btn_loumoscore, &QPushButton::clicked, 
        this, &DlgProductScoreSmartCroppingOfBags::btn_loumoscore_clicked);
    QObject::connect(ui->btn_loumoarea, &QPushButton::clicked,
        this, &DlgProductScoreSmartCroppingOfBags::btn_loumoarea_clicked);

    // 稀疏档
    QObject::connect(ui->ckb_xishudang, &QCheckBox::clicked,
        this, &DlgProductScoreSmartCroppingOfBags::ckb_xishudang_checked);
    QObject::connect(ui->btn_xishudangscore, &QPushButton::clicked, 
        this, &DlgProductScoreSmartCroppingOfBags::btn_xishudangscore_clicked);
    QObject::connect(ui->btn_xishudangarea, &QPushButton::clicked,
        this, &DlgProductScoreSmartCroppingOfBags::btn_xishudangarea_clicked);

    // 二维码
    QObject::connect(ui->ckb_erweima, &QCheckBox::clicked, 
        this, &DlgProductScoreSmartCroppingOfBags::ckb_erweima_checked);
    QObject::connect(ui->btn_erweimascore, &QPushButton::clicked, 
        this, &DlgProductScoreSmartCroppingOfBags::btn_erweimascore_clicked);
    QObject::connect(ui->btn_erweimaarea, &QPushButton::clicked, 
        this, &DlgProductScoreSmartCroppingOfBags::btn_erweimaarea_clicked);

    // 大墨点
    QObject::connect(ui->ckb_damodian, &QCheckBox::clicked, 
        this, &DlgProductScoreSmartCroppingOfBags::ckb_damodian_checked);
    QObject::connect(ui->btn_damodianscore, &QPushButton::clicked, 
        this, &DlgProductScoreSmartCroppingOfBags::btn_damodianscore_clicked);
    QObject::connect(ui->btn_damodianarea, &QPushButton::clicked, 
        this, &DlgProductScoreSmartCroppingOfBags::btn_damodianarea_clicked);

    // 孔洞
    QObject::connect(ui->ckb_kongdong, &QCheckBox::clicked, 
        this, &DlgProductScoreSmartCroppingOfBags::ckb_kongdong_checked);
    QObject::connect(ui->btn_kongdongscore, &QPushButton::clicked, 
        this, &DlgProductScoreSmartCroppingOfBags::btn_kongdongscore_clicked);
    QObject::connect(ui->btn_kongdongarea, &QPushButton::clicked, 
        this, &DlgProductScoreSmartCroppingOfBags::btn_kongdongarea_clicked);

    // 色标
    QObject::connect(ui->ckb_sebiao, &QCheckBox::clicked, 
        this, &DlgProductScoreSmartCroppingOfBags::ckb_sebiao_checked);
    QObject::connect(ui->btn_sebiaoscore, &QPushButton::clicked, 
        this, &DlgProductScoreSmartCroppingOfBags::btn_sebiaoscore_clicked);
    QObject::connect(ui->btn_sebiaoarea, &QPushButton::clicked,
        this, &DlgProductScoreSmartCroppingOfBags::btn_sebiaoarea_clicked);

    // 印刷缺陷
    QObject::connect(ui->ckb_yinshuaquexian, &QCheckBox::clicked, 
        this, &DlgProductScoreSmartCroppingOfBags::ckb_yinshuaquexian_checked);
    QObject::connect(ui->btn_yinshuaquexianscore, &QPushButton::clicked,
        this, &DlgProductScoreSmartCroppingOfBags::btn_yinshuaquexianscore_clicked);
    QObject::connect(ui->btn_yinshuaquexianarea, &QPushButton::clicked,
        this, &DlgProductScoreSmartCroppingOfBags::btn_yinshuaquexianarea_clicked);

    // 小破洞
    QObject::connect(ui->ckb_xiaopodong, &QCheckBox::clicked, 
        this, &DlgProductScoreSmartCroppingOfBags::ckb_xiaopodong_checked);
    QObject::connect(ui->btn_xiaopodongscore, &QPushButton::clicked,
        this, &DlgProductScoreSmartCroppingOfBags::btn_xiaopodongscore_clicked);
    QObject::connect(ui->btn_xiaopodongarea, &QPushButton::clicked, 
        this, &DlgProductScoreSmartCroppingOfBags::btn_xiaopodongarea_clicked);

    // 胶带
    QObject::connect(ui->ckb_jiaodai, &QCheckBox::clicked, 
        this, &DlgProductScoreSmartCroppingOfBags::ckb_jiaodai_checked);
    QObject::connect(ui->btn_jiaodaiscore, &QPushButton::clicked, 
        this, &DlgProductScoreSmartCroppingOfBags::btn_jiaodaiscore_clicked);
    QObject::connect(ui->btn_jiaodaiarea, &QPushButton::clicked, 
        this, &DlgProductScoreSmartCroppingOfBags::btn_jiaodaiarea_clicked);

}

void DlgProductScoreSmartCroppingOfBags::btn_close_clicked()
{
    auto& GlobalStructData = GlobalStructDataSmartCroppingOfBags::getInstance();
    GlobalStructData.saveDlgProductScoreConfig();
    this->close();
}

void DlgProductScoreSmartCroppingOfBags::ckb_heiba_checked()
{
    auto& scoreConfig = GlobalStructDataSmartCroppingOfBags::getInstance().scoreConfig;
    scoreConfig.heiba = ui->ckb_heiba->isChecked();
}

void DlgProductScoreSmartCroppingOfBags::btn_heibascore_clicked()
{
    NumberKeyboard numKeyBord;
    numKeyBord.setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
    auto isAccept = numKeyBord.exec();
    if (isAccept == QDialog::Accepted)
    {
        auto value = numKeyBord.getValue();
        if (value.toDouble() < 0)
        {
            QMessageBox::warning(this, "提示", "请输入大于0的数值");
            return;
        }
        auto& globalStructScoreConfig = GlobalStructDataSmartCroppingOfBags::getInstance().scoreConfig;
        ui->btn_heibascore->setText(value);
        globalStructScoreConfig.heibascore = value.toDouble();
    }
}

void DlgProductScoreSmartCroppingOfBags::btn_heibaarea_clicked()
{
    NumberKeyboard numKeyBord;
    numKeyBord.setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
    auto isAccept = numKeyBord.exec();
    if (isAccept == QDialog::Accepted)
    {
        auto value = numKeyBord.getValue();
        if (value.toDouble() < 0)
        {
            QMessageBox::warning(this, "提示", "请输入大于0的数值");
            return;
        }
        auto& globalStructScoreConfig = GlobalStructDataSmartCroppingOfBags::getInstance().scoreConfig;
        ui->btn_heibaarea->setText(value);
        globalStructScoreConfig.heibaarea = value.toDouble();
    }
}

void DlgProductScoreSmartCroppingOfBags::ckb_shudang_checked()
{
    auto& scoreConfig = GlobalStructDataSmartCroppingOfBags::getInstance().scoreConfig;
    scoreConfig.shudang = ui->ckb_shudang->isChecked();
}

void DlgProductScoreSmartCroppingOfBags::btn_shudangscore_clicked()
{
    NumberKeyboard numKeyBord;
    numKeyBord.setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
    auto isAccept = numKeyBord.exec();
    if (isAccept == QDialog::Accepted)
    {
        auto value = numKeyBord.getValue();
        if (value.toDouble() < 0)
        {
            QMessageBox::warning(this, "提示", "请输入大于0的数值");
            return;
        }
        auto& globalStructScoreConfig = GlobalStructDataSmartCroppingOfBags::getInstance().scoreConfig;
        ui->btn_shudangscore->setText(value);
        globalStructScoreConfig.shudangscore = value.toDouble();
    }
}

void DlgProductScoreSmartCroppingOfBags::btn_shudangarea_clicked()
{
    NumberKeyboard numKeyBord;
    numKeyBord.setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
    auto isAccept = numKeyBord.exec();
    if (isAccept == QDialog::Accepted)
    {
        auto value = numKeyBord.getValue();
        if (value.toDouble() < 0)
        {
            QMessageBox::warning(this, "提示", "请输入大于0的数值");
            return;
        }
        auto& globalStructScoreConfig = GlobalStructDataSmartCroppingOfBags::getInstance().scoreConfig;
        ui->btn_shudangarea->setText(value);
        globalStructScoreConfig.shudangarea = value.toDouble();
    }
}

void DlgProductScoreSmartCroppingOfBags::ckb_huapo_checked()
{
    auto& scoreConfig = GlobalStructDataSmartCroppingOfBags::getInstance().scoreConfig;
    scoreConfig.huapo = ui->ckb_huapo->isChecked();
}

void DlgProductScoreSmartCroppingOfBags::btn_huaposcore_clicked()
{
    NumberKeyboard numKeyBord;
    numKeyBord.setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
    auto isAccept = numKeyBord.exec();
    if (isAccept == QDialog::Accepted)
    {
        auto value = numKeyBord.getValue();
        if (value.toDouble() < 0)
        {
            QMessageBox::warning(this, "提示", "请输入大于0的数值");
            return;
        }
        auto& globalStructScoreConfig = GlobalStructDataSmartCroppingOfBags::getInstance().scoreConfig;
        ui->btn_huaposcore->setText(value);
        globalStructScoreConfig.huaposcore = value.toDouble();
    }
}

void DlgProductScoreSmartCroppingOfBags::btn_huapoarea_clicked()
{
    NumberKeyboard numKeyBord;
    numKeyBord.setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
    auto isAccept = numKeyBord.exec();
    if (isAccept == QDialog::Accepted)
    {
        auto value = numKeyBord.getValue();
        if (value.toDouble() < 0)
        {
            QMessageBox::warning(this, "提示", "请输入大于0的数值");
            return;
        }
        auto& globalStructScoreConfig = GlobalStructDataSmartCroppingOfBags::getInstance().scoreConfig;
        ui->btn_huapoarea->setText(value);
        globalStructScoreConfig.huapoarea = value.toDouble();
    }
}

void DlgProductScoreSmartCroppingOfBags::ckb_jietou_checked()
{
    auto& scoreConfig = GlobalStructDataSmartCroppingOfBags::getInstance().scoreConfig;
    scoreConfig.jietou = ui->ckb_jietou->isChecked();
}

void DlgProductScoreSmartCroppingOfBags::btn_jietouscore_clicked()
{
    NumberKeyboard numKeyBord;
    numKeyBord.setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
    auto isAccept = numKeyBord.exec();
    if (isAccept == QDialog::Accepted)
    {
        auto value = numKeyBord.getValue();
        if (value.toDouble() < 0)
        {
            QMessageBox::warning(this, "提示", "请输入大于0的数值");
            return;
        }
        auto& globalStructScoreConfig = GlobalStructDataSmartCroppingOfBags::getInstance().scoreConfig;
        ui->btn_jietouscore->setText(value);
        globalStructScoreConfig.jietouscore = value.toDouble();
    }
}

void DlgProductScoreSmartCroppingOfBags::btn_jietouarea_clicked()
{
    NumberKeyboard numKeyBord;
    numKeyBord.setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
    auto isAccept = numKeyBord.exec();
    if (isAccept == QDialog::Accepted)
    {
        auto value = numKeyBord.getValue();
        if (value.toDouble() < 0)
        {
            QMessageBox::warning(this, "提示", "请输入大于0的数值");
            return;
        }
        auto& globalStructScoreConfig = GlobalStructDataSmartCroppingOfBags::getInstance().scoreConfig;
        ui->btn_jietouarea->setText(value);
        globalStructScoreConfig.jietouarea = value.toDouble();
    }
}

void DlgProductScoreSmartCroppingOfBags::ckb_guasi_checked()
{
    auto& scoreConfig = GlobalStructDataSmartCroppingOfBags::getInstance().scoreConfig;
    scoreConfig.guasi = ui->ckb_guasi->isChecked();
}

void DlgProductScoreSmartCroppingOfBags::btn_guasiscore_clicked()
{
    NumberKeyboard numKeyBord;
    numKeyBord.setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
    auto isAccept = numKeyBord.exec();
    if (isAccept == QDialog::Accepted)
    {
        auto value = numKeyBord.getValue();
        if (value.toDouble() < 0)
        {
            QMessageBox::warning(this, "提示", "请输入大于0的数值");
            return;
        }
        auto& globalStructScoreConfig = GlobalStructDataSmartCroppingOfBags::getInstance().scoreConfig;
        ui->btn_guasiscore->setText(value);
        globalStructScoreConfig.guasiscore = value.toDouble();
    }
}

void DlgProductScoreSmartCroppingOfBags::btn_guasiarea_clicked()
{
    NumberKeyboard numKeyBord;
    numKeyBord.setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
    auto isAccept = numKeyBord.exec();
    if (isAccept == QDialog::Accepted)
    {
        auto value = numKeyBord.getValue();
        if (value.toDouble() < 0)
        {
            QMessageBox::warning(this, "提示", "请输入大于0的数值");
            return;
        }
        auto& globalStructScoreConfig = GlobalStructDataSmartCroppingOfBags::getInstance().scoreConfig;
        ui->btn_guasiarea->setText(value);
        globalStructScoreConfig.guasiarea = value.toDouble();
    }
}

void DlgProductScoreSmartCroppingOfBags::ckb_podong_checked()
{
    auto& scoreConfig = GlobalStructDataSmartCroppingOfBags::getInstance().scoreConfig;
    scoreConfig.podong = ui->ckb_podong->isChecked();
}

void DlgProductScoreSmartCroppingOfBags::btn_podongscore_clicked()
{
    NumberKeyboard numKeyBord;
    numKeyBord.setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
    auto isAccept = numKeyBord.exec();
    if (isAccept == QDialog::Accepted)
    {
        auto value = numKeyBord.getValue();
        if (value.toDouble() < 0)
        {
            QMessageBox::warning(this, "提示", "请输入大于0的数值");
            return;
        }
        auto& globalStructScoreConfig = GlobalStructDataSmartCroppingOfBags::getInstance().scoreConfig;
        ui->btn_podongscore->setText(value);
        globalStructScoreConfig.podongscore = value.toDouble();
    }
}

void DlgProductScoreSmartCroppingOfBags::btn_podongarea_clicked()
{
    NumberKeyboard numKeyBord;
    numKeyBord.setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
    auto isAccept = numKeyBord.exec();
    if (isAccept == QDialog::Accepted)
    {
        auto value = numKeyBord.getValue();
        if (value.toDouble() < 0)
        {
            QMessageBox::warning(this, "提示", "请输入大于0的数值");
            return;
        }
        auto& globalStructScoreConfig = GlobalStructDataSmartCroppingOfBags::getInstance().scoreConfig;
        ui->btn_podongarea->setText(value);
        globalStructScoreConfig.podongarea = value.toDouble();
    }
}

void DlgProductScoreSmartCroppingOfBags::ckb_zangwu_checked()
{
    auto& scoreConfig = GlobalStructDataSmartCroppingOfBags::getInstance().scoreConfig;
    scoreConfig.zangwu = ui->ckb_zangwu->isChecked();
}

void DlgProductScoreSmartCroppingOfBags::btn_zangwuscore_clicked()
{
    NumberKeyboard numKeyBord;
    numKeyBord.setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
    auto isAccept = numKeyBord.exec();
    if (isAccept == QDialog::Accepted)
    {
        auto value = numKeyBord.getValue();
        if (value.toDouble() < 0)
        {
            QMessageBox::warning(this, "提示", "请输入大于0的数值");
            return;
        }
        auto& globalStructScoreConfig = GlobalStructDataSmartCroppingOfBags::getInstance().scoreConfig;
        ui->btn_zangwuscore->setText(value);
        globalStructScoreConfig.zangwuscore = value.toDouble();
    }
}

void DlgProductScoreSmartCroppingOfBags::btn_zangwuarea_clicked()
{
    NumberKeyboard numKeyBord;
    numKeyBord.setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
    auto isAccept = numKeyBord.exec();
    if (isAccept == QDialog::Accepted)
    {
        auto value = numKeyBord.getValue();
        if (value.toDouble() < 0)
        {
            QMessageBox::warning(this, "提示", "请输入大于0的数值");
            return;
        }
        auto& globalStructScoreConfig = GlobalStructDataSmartCroppingOfBags::getInstance().scoreConfig;
        ui->btn_zangwuarea->setText(value);
        globalStructScoreConfig.zangwuarea = value.toDouble();
    }
}

void DlgProductScoreSmartCroppingOfBags::ckb_noshudang_checked()
{
    auto& scoreConfig = GlobalStructDataSmartCroppingOfBags::getInstance().scoreConfig;
    scoreConfig.noshudang = ui->ckb_noshudang->isChecked();
}

void DlgProductScoreSmartCroppingOfBags::btn_noshudangscore_clicked()
{
    NumberKeyboard numKeyBord;
    numKeyBord.setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
    auto isAccept = numKeyBord.exec();
    if (isAccept == QDialog::Accepted)
    {
        auto value = numKeyBord.getValue();
        if (value.toDouble() < 0)
        {
            QMessageBox::warning(this, "提示", "请输入大于0的数值");
            return;
        }
        auto& globalStructScoreConfig = GlobalStructDataSmartCroppingOfBags::getInstance().scoreConfig;
        ui->btn_noshudangscore->setText(value);
        globalStructScoreConfig.noshudangscore = value.toDouble();
    }
}

void DlgProductScoreSmartCroppingOfBags::btn_noshudangarea_clicked()
{
    NumberKeyboard numKeyBord;
    numKeyBord.setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
    auto isAccept = numKeyBord.exec();
    if (isAccept == QDialog::Accepted)
    {
        auto value = numKeyBord.getValue();
        if (value.toDouble() < 0)
        {
            QMessageBox::warning(this, "提示", "请输入大于0的数值");
            return;
        }
        auto& globalStructScoreConfig = GlobalStructDataSmartCroppingOfBags::getInstance().scoreConfig;
        ui->btn_noshudangarea->setText(value);
        globalStructScoreConfig.noshudangarea = value.toDouble();
    }
}

void DlgProductScoreSmartCroppingOfBags::ckb_modian_checked()
{
    auto& scoreConfig = GlobalStructDataSmartCroppingOfBags::getInstance().scoreConfig;
    scoreConfig.modian = ui->ckb_modian->isChecked();
}

void DlgProductScoreSmartCroppingOfBags::btn_modianscore_clicked()
{
    NumberKeyboard numKeyBord;
    numKeyBord.setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
    auto isAccept = numKeyBord.exec();
    if (isAccept == QDialog::Accepted)
    {
        auto value = numKeyBord.getValue();
        if (value.toDouble() < 0)
        {
            QMessageBox::warning(this, "提示", "请输入大于0的数值");
            return;
        }
        auto& globalStructScoreConfig = GlobalStructDataSmartCroppingOfBags::getInstance().scoreConfig;
        ui->btn_modianscore->setText(value);
        globalStructScoreConfig.modianscore = value.toDouble();
    }
}

void DlgProductScoreSmartCroppingOfBags::btn_modianarea_clicked()
{
    NumberKeyboard numKeyBord;
    numKeyBord.setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
    auto isAccept = numKeyBord.exec();
    if (isAccept == QDialog::Accepted)
    {
        auto value = numKeyBord.getValue();
        if (value.toDouble() < 0)
        {
            QMessageBox::warning(this, "提示", "请输入大于0的数值");
            return;
        }
        auto& globalStructScoreConfig = GlobalStructDataSmartCroppingOfBags::getInstance().scoreConfig;
        ui->btn_modianarea->setText(value);
        globalStructScoreConfig.modianarea = value.toDouble();
    }
}

void DlgProductScoreSmartCroppingOfBags::ckb_loumo_checked()
{
    auto& scoreConfig = GlobalStructDataSmartCroppingOfBags::getInstance().scoreConfig;
    scoreConfig.loumo = ui->ckb_loumo->isChecked();
}

void DlgProductScoreSmartCroppingOfBags::btn_loumoscore_clicked()
{
    NumberKeyboard numKeyBord;
    numKeyBord.setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
    auto isAccept = numKeyBord.exec();
    if (isAccept == QDialog::Accepted)
    {
        auto value = numKeyBord.getValue();
        if (value.toDouble() < 0)
        {
            QMessageBox::warning(this, "提示", "请输入大于0的数值");
            return;
        }
        auto& globalStructScoreConfig = GlobalStructDataSmartCroppingOfBags::getInstance().scoreConfig;
        ui->btn_loumoscore->setText(value);
        globalStructScoreConfig.loumoscore = value.toDouble();
    }
}

void DlgProductScoreSmartCroppingOfBags::btn_loumoarea_clicked()
{
    NumberKeyboard numKeyBord;
    numKeyBord.setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
    auto isAccept = numKeyBord.exec();
    if (isAccept == QDialog::Accepted)
    {
        auto value = numKeyBord.getValue();
        if (value.toDouble() < 0)
        {
            QMessageBox::warning(this, "提示", "请输入大于0的数值");
            return;
        }
        auto& globalStructScoreConfig = GlobalStructDataSmartCroppingOfBags::getInstance().scoreConfig;
        ui->btn_loumoarea->setText(value);
        globalStructScoreConfig.loumoarea = value.toDouble();
    }
}

void DlgProductScoreSmartCroppingOfBags::ckb_xishudang_checked()
{
    auto& scoreConfig = GlobalStructDataSmartCroppingOfBags::getInstance().scoreConfig;
    scoreConfig.xishudang = ui->ckb_xishudang->isChecked();
}

void DlgProductScoreSmartCroppingOfBags::btn_xishudangscore_clicked()
{
    NumberKeyboard numKeyBord;
    numKeyBord.setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
    auto isAccept = numKeyBord.exec();
    if (isAccept == QDialog::Accepted)
    {
        auto value = numKeyBord.getValue();
        if (value.toDouble() < 0)
        {
            QMessageBox::warning(this, "提示", "请输入大于0的数值");
            return;
        }
        auto& globalStructScoreConfig = GlobalStructDataSmartCroppingOfBags::getInstance().scoreConfig;
        ui->btn_xishudangscore->setText(value);
        globalStructScoreConfig.xishudangscore = value.toDouble();
    }
}

void DlgProductScoreSmartCroppingOfBags::btn_xishudangarea_clicked()
{
    NumberKeyboard numKeyBord;
    numKeyBord.setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
    auto isAccept = numKeyBord.exec();
    if (isAccept == QDialog::Accepted)
    {
        auto value = numKeyBord.getValue();
        if (value.toDouble() < 0)
        {
            QMessageBox::warning(this, "提示", "请输入大于0的数值");
            return;
        }
        auto& globalStructScoreConfig = GlobalStructDataSmartCroppingOfBags::getInstance().scoreConfig;
        ui->btn_xishudangarea->setText(value);
        globalStructScoreConfig.xishudangarea = value.toDouble();
    }
}

void DlgProductScoreSmartCroppingOfBags::ckb_erweima_checked()
{
    auto& scoreConfig = GlobalStructDataSmartCroppingOfBags::getInstance().scoreConfig;
    scoreConfig.erweima = ui->ckb_erweima->isChecked();
}

void DlgProductScoreSmartCroppingOfBags::btn_erweimascore_clicked()
{
    NumberKeyboard numKeyBord;
    numKeyBord.setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
    auto isAccept = numKeyBord.exec();
    if (isAccept == QDialog::Accepted)
    {
        auto value = numKeyBord.getValue();
        if (value.toDouble() < 0)
        {
            QMessageBox::warning(this, "提示", "请输入大于0的数值");
            return;
        }
        auto& globalStructScoreConfig = GlobalStructDataSmartCroppingOfBags::getInstance().scoreConfig;
        ui->btn_erweimascore->setText(value);
        globalStructScoreConfig.erweimascore = value.toDouble();
    }
}

void DlgProductScoreSmartCroppingOfBags::btn_erweimaarea_clicked()
{
    NumberKeyboard numKeyBord;
    numKeyBord.setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
    auto isAccept = numKeyBord.exec();
    if (isAccept == QDialog::Accepted)
    {
        auto value = numKeyBord.getValue();
        if (value.toDouble() < 0)
        {
            QMessageBox::warning(this, "提示", "请输入大于0的数值");
            return;
        }
        auto& globalStructScoreConfig = GlobalStructDataSmartCroppingOfBags::getInstance().scoreConfig;
        ui->btn_erweimaarea->setText(value);
        globalStructScoreConfig.erweimaarea = value.toDouble();
    }
}

void DlgProductScoreSmartCroppingOfBags::ckb_damodian_checked()
{
    auto& scoreConfig = GlobalStructDataSmartCroppingOfBags::getInstance().scoreConfig;
    scoreConfig.damodian = ui->ckb_damodian->isChecked();
}

void DlgProductScoreSmartCroppingOfBags::btn_damodianscore_clicked()
{
    NumberKeyboard numKeyBord;
    numKeyBord.setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
    auto isAccept = numKeyBord.exec();
    if (isAccept == QDialog::Accepted)
    {
        auto value = numKeyBord.getValue();
        if (value.toDouble() < 0)
        {
            QMessageBox::warning(this, "提示", "请输入大于0的数值");
            return;
        }
        auto& globalStructScoreConfig = GlobalStructDataSmartCroppingOfBags::getInstance().scoreConfig;
        ui->btn_damodianscore->setText(value);
        globalStructScoreConfig.damodianscore = value.toDouble();
    }
}

void DlgProductScoreSmartCroppingOfBags::btn_damodianarea_clicked()
{
    NumberKeyboard numKeyBord;
    numKeyBord.setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
    auto isAccept = numKeyBord.exec();
    if (isAccept == QDialog::Accepted)
    {
        auto value = numKeyBord.getValue();
        if (value.toDouble() < 0)
        {
            QMessageBox::warning(this, "提示", "请输入大于0的数值");
            return;
        }
        auto& globalStructScoreConfig = GlobalStructDataSmartCroppingOfBags::getInstance().scoreConfig;
        ui->btn_damodianarea->setText(value);
        globalStructScoreConfig.damodianarea = value.toDouble();
    }
}

void DlgProductScoreSmartCroppingOfBags::ckb_kongdong_checked()
{
    auto& scoreConfig = GlobalStructDataSmartCroppingOfBags::getInstance().scoreConfig;
    scoreConfig.kongdong = ui->ckb_kongdong->isChecked();
}

void DlgProductScoreSmartCroppingOfBags::btn_kongdongscore_clicked()
{
    NumberKeyboard numKeyBord;
    numKeyBord.setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
    auto isAccept = numKeyBord.exec();
    if (isAccept == QDialog::Accepted)
    {
        auto value = numKeyBord.getValue();
        if (value.toDouble() < 0)
        {
            QMessageBox::warning(this, "提示", "请输入大于0的数值");
            return;
        }
        auto& globalStructScoreConfig = GlobalStructDataSmartCroppingOfBags::getInstance().scoreConfig;
        ui->btn_kongdongscore->setText(value);
        globalStructScoreConfig.kongdongscore = value.toDouble();
    }
}

void DlgProductScoreSmartCroppingOfBags::btn_kongdongarea_clicked()
{
    NumberKeyboard numKeyBord;
    numKeyBord.setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
    auto isAccept = numKeyBord.exec();
    if (isAccept == QDialog::Accepted)
    {
        auto value = numKeyBord.getValue();
        if (value.toDouble() < 0)
        {
            QMessageBox::warning(this, "提示", "请输入大于0的数值");
            return;
        }
        auto& globalStructScoreConfig = GlobalStructDataSmartCroppingOfBags::getInstance().scoreConfig;
        ui->btn_kongdongarea->setText(value);
        globalStructScoreConfig.kongdongarea = value.toDouble();
    }
}

void DlgProductScoreSmartCroppingOfBags::ckb_sebiao_checked()
{
    auto& scoreConfig = GlobalStructDataSmartCroppingOfBags::getInstance().scoreConfig;
    scoreConfig.sebiao = ui->ckb_sebiao->isChecked();
}

void DlgProductScoreSmartCroppingOfBags::btn_sebiaoscore_clicked()
{
    NumberKeyboard numKeyBord;
    numKeyBord.setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
    auto isAccept = numKeyBord.exec();
    if (isAccept == QDialog::Accepted)
    {
        auto value = numKeyBord.getValue();
        if (value.toDouble() < 0)
        {
            QMessageBox::warning(this, "提示", "请输入大于0的数值");
            return;
        }
        auto& globalStructScoreConfig = GlobalStructDataSmartCroppingOfBags::getInstance().scoreConfig;
        ui->btn_sebiaoscore->setText(value);
        globalStructScoreConfig.sebiaoscore = value.toDouble();
    }
}

void DlgProductScoreSmartCroppingOfBags::btn_sebiaoarea_clicked()
{
    NumberKeyboard numKeyBord;
    numKeyBord.setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
    auto isAccept = numKeyBord.exec();
    if (isAccept == QDialog::Accepted)
    {
        auto value = numKeyBord.getValue();
        if (value.toDouble() < 0)
        {
            QMessageBox::warning(this, "提示", "请输入大于0的数值");
            return;
        }
        auto& globalStructScoreConfig = GlobalStructDataSmartCroppingOfBags::getInstance().scoreConfig;
        ui->btn_sebiaoarea->setText(value);
        globalStructScoreConfig.sebiaoarea = value.toDouble();
    }
}

void DlgProductScoreSmartCroppingOfBags::ckb_yinshuaquexian_checked()
{
    auto& scoreConfig = GlobalStructDataSmartCroppingOfBags::getInstance().scoreConfig;
    scoreConfig.yinshuaquexian = ui->ckb_yinshuaquexian->isChecked();
}

void DlgProductScoreSmartCroppingOfBags::btn_yinshuaquexianscore_clicked()
{
    NumberKeyboard numKeyBord;
    numKeyBord.setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
    auto isAccept = numKeyBord.exec();
    if (isAccept == QDialog::Accepted)
    {
        auto value = numKeyBord.getValue();
        if (value.toDouble() < 0)
        {
            QMessageBox::warning(this, "提示", "请输入大于0的数值");
            return;
        }
        auto& globalStructScoreConfig = GlobalStructDataSmartCroppingOfBags::getInstance().scoreConfig;
        ui->btn_yinshuaquexianscore->setText(value);
        globalStructScoreConfig.yinshuaquexianscore = value.toDouble();
    }
}

void DlgProductScoreSmartCroppingOfBags::btn_yinshuaquexianarea_clicked()
{
    NumberKeyboard numKeyBord;
    numKeyBord.setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
    auto isAccept = numKeyBord.exec();
    if (isAccept == QDialog::Accepted)
    {
        auto value = numKeyBord.getValue();
        if (value.toDouble() < 0)
        {
            QMessageBox::warning(this, "提示", "请输入大于0的数值");
            return;
        }
        auto& globalStructScoreConfig = GlobalStructDataSmartCroppingOfBags::getInstance().scoreConfig;
        ui->btn_yinshuaquexianarea->setText(value);
        globalStructScoreConfig.yinshuaquexianarea = value.toDouble();
    }
}

void DlgProductScoreSmartCroppingOfBags::ckb_xiaopodong_checked()
{
    auto& scoreConfig = GlobalStructDataSmartCroppingOfBags::getInstance().scoreConfig;
    scoreConfig.xiaopodong = ui->ckb_xiaopodong->isChecked();
}

void DlgProductScoreSmartCroppingOfBags::btn_xiaopodongscore_clicked()
{
    NumberKeyboard numKeyBord;
    numKeyBord.setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
    auto isAccept = numKeyBord.exec();
    if (isAccept == QDialog::Accepted)
    {
        auto value = numKeyBord.getValue();
        if (value.toDouble() < 0)
        {
            QMessageBox::warning(this, "提示", "请输入大于0的数值");
            return;
        }
        auto& globalStructScoreConfig = GlobalStructDataSmartCroppingOfBags::getInstance().scoreConfig;
        ui->btn_xiaopodongscore->setText(value);
        globalStructScoreConfig.xiaopodongscore = value.toDouble();
    }
}

void DlgProductScoreSmartCroppingOfBags::btn_xiaopodongarea_clicked()
{
    NumberKeyboard numKeyBord;
    numKeyBord.setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
    auto isAccept = numKeyBord.exec();
    if (isAccept == QDialog::Accepted)
    {
        auto value = numKeyBord.getValue();
        if (value.toDouble() < 0)
        {
            QMessageBox::warning(this, "提示", "请输入大于0的数值");
            return;
        }
        auto& globalStructScoreConfig = GlobalStructDataSmartCroppingOfBags::getInstance().scoreConfig;
        ui->btn_xiaopodongarea->setText(value);
        globalStructScoreConfig.xiaopodongarea = value.toDouble();
    }
}

void DlgProductScoreSmartCroppingOfBags::ckb_jiaodai_checked()
{
    auto& scoreConfig = GlobalStructDataSmartCroppingOfBags::getInstance().scoreConfig;
    scoreConfig.jiaodai = ui->ckb_jiaodai->isChecked();
}

void DlgProductScoreSmartCroppingOfBags::btn_jiaodaiscore_clicked()
{
    NumberKeyboard numKeyBord;
    numKeyBord.setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
    auto isAccept = numKeyBord.exec();
    if (isAccept == QDialog::Accepted)
    {
        auto value = numKeyBord.getValue();
        if (value.toDouble() < 0)
        {
            QMessageBox::warning(this, "提示", "请输入大于0的数值");
            return;
        }
        auto& globalStructScoreConfig = GlobalStructDataSmartCroppingOfBags::getInstance().scoreConfig;
        ui->btn_jiaodaiscore->setText(value);
        globalStructScoreConfig.jiaodaiscore = value.toDouble();
    }
}

void DlgProductScoreSmartCroppingOfBags::btn_jiaodaiarea_clicked()
{
    NumberKeyboard numKeyBord;
    numKeyBord.setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
    auto isAccept = numKeyBord.exec();
    if (isAccept == QDialog::Accepted)
    {
        auto value = numKeyBord.getValue();
        if (value.toDouble() < 0)
        {
            QMessageBox::warning(this, "提示", "请输入大于0的数值");
            return;
        }
        auto& globalStructScoreConfig = GlobalStructDataSmartCroppingOfBags::getInstance().scoreConfig;
        ui->btn_jiaodaiarea->setText(value);
        globalStructScoreConfig.jiaodaiarea = value.toDouble();
    }
}