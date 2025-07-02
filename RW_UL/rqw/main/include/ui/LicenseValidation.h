#pragma once

#include <QtWidgets/QDialog>

QT_BEGIN_NAMESPACE
namespace Ui { class LicenseValidationClass; };
QT_END_NAMESPACE

class LicenseValidation : public QDialog
{
    Q_OBJECT

public:
    LicenseValidation(QWidget *parent = nullptr);
    ~LicenseValidation();

private:
    Ui::LicenseValidationClass *ui;
};
