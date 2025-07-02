#include "LicenseValidation.h"

#include "ui_LicenseValidation.h"

LicenseValidation::LicenseValidation(QWidget *parent)
    : QDialog(parent)
    , ui(new Ui::LicenseValidationClass())
{
    ui->setupUi(this);
}

LicenseValidation::~LicenseValidation()
{
    delete ui;
}
