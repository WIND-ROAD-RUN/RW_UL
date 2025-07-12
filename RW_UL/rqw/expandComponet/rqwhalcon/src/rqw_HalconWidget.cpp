#include"rqw_HalconWidget.hpp"
#include <QResizeEvent>
#include <unordered_set>

#include "halconcpp/HalconCpp.h"

#include"rqw_HalconUtilty.hpp"

namespace rw {
	namespace rqw
	{
		HalconWidgetDisObject::HalconWidgetDisObject(HalconCpp::HObject* obj)
			:_object(obj)
		{

		}

		HalconWidgetDisObject::HalconWidgetDisObject(const HalconCpp::HImage& image)
			:type(HalconWidgetDisObject::ObjectType::Image)
		{
            HalconCpp::HImage* newImage = new HalconCpp::HImage(image);
            _object = newImage;
		}

		HalconWidgetDisObject::HalconWidgetDisObject(const cv::Mat& mat)
            :type(HalconWidgetDisObject::ObjectType::Image)
		{
            HalconCpp::HImage hImage = CvMatToHImage(mat);
            auto newImage = new HalconCpp::HImage(hImage);
            _object = newImage;
		}

		HalconWidgetDisObject::HalconWidgetDisObject(const QImage& image)
            :type(HalconWidgetDisObject::ObjectType::Image)
		{
            HalconCpp::HImage hImage = QImageToHImage(image);
            auto newImage = new HalconCpp::HImage(hImage);
			_object = newImage;
		}

		HalconWidgetDisObject::HalconWidgetDisObject(const QPixmap& pixmap)
            :type(HalconWidgetDisObject::ObjectType::Image)
		{
            QImage image = pixmap.toImage();
            HalconCpp::HImage hImage = QImageToHImage(image);
            auto newImage = new HalconCpp::HImage(hImage);
            _object = newImage;
		}

		HalconWidgetDisObject::~HalconWidgetDisObject()
		{
            if (_object)
            {
                delete _object;
            }
		}

		HalconWidgetDisObject::HalconWidgetDisObject(const HalconWidgetDisObject& other)
            : _object(other._object ? new HalconCpp::HObject(*other._object) : nullptr),
            id(other.id),
            descrption(other.descrption),
            isShow(other.isShow),
			type(other.type),
            painterConfig(other.painterConfig)
        {
        }

		HalconWidgetDisObject::HalconWidgetDisObject(HalconWidgetDisObject&& other) noexcept
            : _object(other._object),
            id(other.id),
            descrption(std::move(other.descrption)),
            isShow(other.isShow),
			type(other.type),
            painterConfig(other.painterConfig)
        {
            other._object = nullptr;
        }

		HalconWidgetDisObject& HalconWidgetDisObject::operator=(const HalconWidgetDisObject& other)
        {
            if (this != &other)
            {
                // 释放当前对象
                delete _object;

                // 深拷贝
                _object = other._object ? new HalconCpp::HObject(*other._object) : nullptr;
                id = other.id;
                descrption = other.descrption;
                isShow = other.isShow;
				type = other.type;
				painterConfig = other.painterConfig;
            }
            return *this;
        }

		HalconWidgetDisObject& HalconWidgetDisObject::operator=(HalconWidgetDisObject&& other) noexcept
        {
            if (this != &other)
            {
                // 释放当前对象
                delete _object;

                // 移动资源
                _object = other._object;
                id = other.id;
                descrption = std::move(other.descrption);
                isShow = other.isShow;
				type = other.type;
				painterConfig = other.painterConfig;

                // 清空源对象
                other._object = nullptr;
            }
            return *this;
        }

		bool HalconWidgetDisObject::has_value()
		{
			return _object != nullptr && _object->IsInitialized();
		}

		HalconCpp::HObject* HalconWidgetDisObject::value()
		{
            if (!has_value())
            {
                throw std::runtime_error("HalconWidgetDisObject does not contain a valid HObject.");
            }
			return _object;
		}

		void HalconWidgetDisObject::release()
		{
            if (_object)
            {
                delete _object;
                _object = nullptr;
                type = ObjectType::Undefined;
			}
		}

		void HalconWidgetDisObject::updateObject(const HalconCpp::HObject& object)
		{
            if (_object)
            {
                delete _object; 
            }
			_object = new HalconCpp::HObject(object); 
		}

		void HalconWidgetDisObject::updateObject(HalconCpp::HObject* object)
		{
            if (_object)
            {
                delete _object; 
            }
			_object = object; 
		}

		HalconWidget::HalconWidget(QWidget* parent)
            : QWidget(parent)
        {
            initialize_halconWindow();
        }

        HalconWidget::~HalconWidget()
        {
            clearHObject();
            close_halconWindow();
        }

        HalconCpp::HTuple* HalconWidget::Handle()
        {
            if (_halconWindowHandle == nullptr)
            {
                initialize_halconWindow();
            }
			return _halconWindowHandle;
        }

        void HalconWidget::appendHObject(const HalconWidgetDisObject& object)
        {
            if (object.id<0)
            {
				throw std::runtime_error("Object ID must be a non-negative integer.");
            }

            for (const auto& existingObject : _halconObjects)
            {
                if (existingObject->id == object.id)
                {
                    throw std::runtime_error("An object with the same ID already exists.");
                }
            }

            HalconWidgetDisObject* newObject = new HalconWidgetDisObject(object);
            _halconObjects.push_back(newObject);
            refresh_allObject();
        }

        void HalconWidget::appendHObject(HalconWidgetDisObject* object)
        {
            if (object->id < 0)
            {
                throw std::runtime_error("Object ID must be a non-negative integer.");
            }

            if (object == nullptr)
            {
                return;
            }
            
            for (const auto& existingObject : _halconObjects)
            {
                if (existingObject->id == object->id)
                {
                    throw std::runtime_error("An object with the same ID already exists.");
                }
            }
            _halconObjects.push_back(object);
			refresh_allObject();
        }


        void HalconWidget::clearHObject()
        {
            _halconObjects.clear();
            if (_halconWindowHandle)
            {
                ClearWindow(*_halconWindowHandle);
			}
        }

        size_t HalconWidget::width()
        {
            if (!_halconWindowHandle)
            {
                return 0; // 如果窗口句柄未初始化，返回 0
            }

            HalconCpp::HTuple row1, col1, row2, col2;
            GetPart(*_halconWindowHandle, &row1, &col1, &row2, &col2);

            // 计算宽度
            return static_cast<size_t>(col2.D() - col1.D() + 1);
        }

        size_t HalconWidget::height()
        {
            if (!_halconWindowHandle)
            {
                return 0; // 如果窗口句柄未初始化，返回 0
            }

            HalconCpp::HTuple row1, col1, row2, col2;
            GetPart(*_halconWindowHandle, &row1, &col1, &row2, &col2);

            // 计算高度
            return static_cast<size_t>(row2.D() - row1.D() + 1);
        }

        HalconWidgetDisObject* HalconWidget::getObjectPtrById(int id)
        {
            for (auto& object : _halconObjects)
            {
                if (object->id == id)
                {
                    return object;
                }
            }
            return new HalconWidgetDisObject(nullptr);
        }

        HalconWidgetDisObject HalconWidget::getObjectById(int id)
        {
            for (auto& object : _halconObjects)
            {
                if (object->id == id)
                {
                    return HalconWidgetDisObject(*object);
                }
            }
            return HalconWidgetDisObject(nullptr);
        }

        bool HalconWidget::eraseObjectById(int id)
        {
            for (auto it = _halconObjects.begin(); it != _halconObjects.end(); ++it)
            {
                if ((*it)->id == id)
                {
                    delete *it; 
                    _halconObjects.erase(it); 
                    refresh_allObject(); 
                    return true; 
                }
            }
			return false; // 未找到对象
        }

        bool HalconWidget::eraseObjectsByType(HalconWidgetDisObject::ObjectType objectType)
        {
            auto ids=getIdsByType(objectType);
            if (ids.empty())
            {
                return false; // 没有找到指定类型的对象
			}
            for (auto it = _halconObjects.begin(); it != _halconObjects.end();)
            {
                if ((*it)->type == objectType)
                {
                    delete *it; // 释放对象内存
                    it = _halconObjects.erase(it); // 删除对象并更新迭代器
                }
                else
                {
                    ++it; // 继续迭代
                }
            }
            refresh_allObject(); // 刷新显示
			return true; // 成功删除指定类型的对象
        }

        std::vector<HalconWidgetDisObjectId> HalconWidget::getAllIds() const
        {
            std::vector<HalconWidgetDisObjectId> ids;
            for (const auto& object : _halconObjects)
            {
                ids.push_back(object->id);
            }
			return ids;
        }

        std::vector<HalconWidgetDisObjectId> HalconWidget::getIdsByType(const HalconWidgetDisObject::ObjectType objectType) const
        {
            std::vector<HalconWidgetDisObjectId> ids;
            for (const auto& object : _halconObjects)
            {
                if (object->type==objectType)
                {
                    ids.push_back(object->id);
                }
            }
            return ids;
        }

        HalconWidgetDisObjectId HalconWidget::getVailidAppendId()
        {
            std::unordered_set<HalconWidgetDisObjectId> existingIds;
            for (const auto& object : _halconObjects)
            {
                existingIds.insert(object->id);
            }

            HalconWidgetDisObjectId newId = 0;
            while (existingIds.find(newId) != existingIds.end())
            {
                ++newId;
            }

            return newId;
        }

        void HalconWidget::refresh_allObject()
        {
            if (!_halconWindowHandle)
            {
                return;
            }

            // 清除窗口内容
            ClearWindow(*_halconWindowHandle);

            if (_halconObjects.empty())
            {
                return; // 如果没有对象需要显示，直接返回
            }

            // 获取窗口大小
            auto size = this->size();
            int windowWidth = size.width();
            int windowHeight = size.height();

            // 获取第一个对象的大小（假设所有对象的大小一致）
            auto ids=getIdsByType(HalconWidgetDisObject::ObjectType::Image);
            if (ids.empty())
            {
                return;
            }
            HalconCpp::HTuple width, height;
            GetImageSize(*getObjectPtrById(ids.front())->value(), &width, &height);

            // 计算图像和窗口的宽高比
            double imgAspectRatio = static_cast<double>(width.I()) / height.I();
            double windowAspectRatio = static_cast<double>(windowWidth) / windowHeight;

            // 设置 Halcon 窗口铺满整个 QWidget
            SetWindowExtents(*_halconWindowHandle, 0, 0, windowWidth, windowHeight);

            // 计算显示区域
            int partWidth, partHeight, offsetX, offsetY;
            if (imgAspectRatio > windowAspectRatio)
            {
                // 图像更宽，以宽度为基准
                partWidth = width.I();
                partHeight = static_cast<int>(width.I() / windowAspectRatio);
                offsetX = 0;
                offsetY = (partHeight - height.I()) / 2;
            }
            else
            {
                // 图像更高，以高度为基准
                partHeight = height.I();
                partWidth = static_cast<int>(height.I() * windowAspectRatio);
                offsetX = (partWidth - width.I()) / 2;
                offsetY = 0;
            }

            // 设置显示区域，使图像居中等比例显示
            SetPart(*_halconWindowHandle, -offsetY, -offsetX, height.I() - 1 + offsetY, width.I() - 1 + offsetX);

            // 显示所有对象
            for (auto& object : _halconObjects)
            {
                if (object->isShow)
                {
                    display_HalconWidgetDisObject(object);
                }
            }
        }

        void HalconWidget::updateWidget()
        {
            refresh_allObject();
        }


        void HalconWidget::appendVerticalLine(int position, const PainterConfig& config)
        {
            if (!_halconWindowHandle)
            {
                return;
            }

            // 获取窗口的高度和宽度范围
            HalconCpp::HTuple row1, col1, row2, col2;
            GetPart(*_halconWindowHandle, &row1, &col1, &row2, &col2);

            // 校准 position，使其基于窗口的坐标范围
            int calibratedPosition = static_cast<int>(col1.D() + position);

            // 检查校准后的 position 是否在有效范围内
            if (calibratedPosition - config.thickness / 2 < col1.D() || calibratedPosition + config.thickness / 2 > col2.D())
            {
                throw std::out_of_range("Position is out of the valid range for the vertical line.");
            }

            // 生成垂直线
            auto verticalLine = new HalconCpp::HObject;
            HalconCpp::GenRectangle1(verticalLine, row1.D(), calibratedPosition - config.thickness / 2, row2.D(), calibratedPosition + config.thickness / 2);

            // 设置颜色
            auto [r, g, b] = RQWColorToRGB(config.color);
            SetRgb(*_halconWindowHandle, r, g, b);

            // 创建并添加对象
            HalconWidgetDisObject object(verticalLine);
            object.isShow = true;
            object.painterConfig = config;
            object.type = HalconWidgetDisObject::ObjectType::Line;
            object.id = getVailidAppendId();
            appendHObject(object);
        }

        void HalconWidget::appendHorizontalLine(int position, const PainterConfig& config)
        {
            if (!_halconWindowHandle)
            {
                return;
            }

            HalconCpp::HTuple row1, col1, row2, col2;
            GetPart(*_halconWindowHandle, &row1, &col1, &row2, &col2);

            int calibratedPosition = static_cast<int>(row1.D() + position);

            if (calibratedPosition - config.thickness / 2 < row1.D() || calibratedPosition + config.thickness / 2 > row2.D())
            {
                throw std::out_of_range("Position is out of the valid range for the horizontal line.");
            }

            auto horizontalLine = new HalconCpp::HObject;
            HalconCpp::GenRectangle1(horizontalLine, calibratedPosition - config.thickness / 2, col1.D(), calibratedPosition + config.thickness / 2, col2.D());

            auto [r,g,b] = RQWColorToRGB(config.color);
            SetRgb(*_halconWindowHandle, r, g, b);

            HalconWidgetDisObject object(horizontalLine);
            object.isShow = true;
            object.painterConfig = config;
            object.type = HalconWidgetDisObject::ObjectType::Line;
            object.id = getVailidAppendId();
            appendHObject(object);
        }

        bool HalconWidget::setObjectVisible(const HalconWidgetDisObjectId id, const bool visible)
        {
            auto object=getObjectPtrById(id);
            if (object == nullptr || !object->has_value())
            {
                return false; 
			}
            object->isShow = visible; 
            refresh_allObject(); 
			return true; 
        }

        bool HalconWidget::isDrawing()
        {
            return _isDrawingRect;
        }


        void HalconWidget::initialize_halconWindow()
        {
            _halconWindowHandle = new HalconCpp::HTuple();
            Hlong winId = this->winId();
            auto size = this->size();

            OpenWindow(0, 0, size.width(), size.height(), winId, "visible", "", _halconWindowHandle);
        }

        void HalconWidget::close_halconWindow()
        {
            if (_halconWindowHandle != nullptr)
            {
                CloseWindow(*_halconWindowHandle);
                delete _halconWindowHandle;
            }
        }

        void HalconWidget::wheelEvent(QWheelEvent* event)
        {
            if (_isDrawingRect) {
                event->ignore(); 
                return;
            }

            if (rect().contains(event->position().toPoint())) { 
                int delta = event->angleDelta().y(); 
                double scaleFactor = (delta > 0) ? 1.1 : 0.9; 

                QPointF mousePos = event->position();
                int mouseX = static_cast<int>(mousePos.x());
                int mouseY = static_cast<int>(mousePos.y());

                HalconCpp::HTuple row1, col1, row2, col2;
                GetPart(*_halconWindowHandle, &row1, &col1, &row2, &col2);

                double currentWidth = col2.D() - col1.D() + 1;
                double currentHeight = row2.D() - row1.D() + 1;

                double relativeX = col1.D() + (mouseX / static_cast<double>(width())) * currentWidth;
                double relativeY = row1.D() + (mouseY / static_cast<double>(height())) * currentHeight;

                double newWidth = currentWidth / scaleFactor;
                double newHeight = currentHeight / scaleFactor;
                double newCol1 = relativeX - (mouseX / static_cast<double>(width())) * newWidth;
                double newRow1 = relativeY - (mouseY / static_cast<double>(height())) * newHeight;
                double newCol2 = newCol1 + newWidth - 1;
                double newRow2 = newRow1 + newHeight - 1;

                ClearWindow(*_halconWindowHandle);

                SetPart(*_halconWindowHandle, newRow1, newCol1, newRow2, newCol2);

                for (auto& object : _halconObjects)
                {
                    if (object->isShow)
                    {
                        display_HalconWidgetDisObject(object);
                    }
                }

                event->accept(); 
            }
            else {
                event->ignore(); 
            }
        }

        void HalconWidget::showEvent(QShowEvent* event)
        {
            refresh_allObject();
        }

        void HalconWidget::resizeEvent(QResizeEvent* event)
        {
            // 定义最小宽度和高度
            const int minWidth = 100;  // 最小宽度
            const int minHeight = 100; // 最小高度

            // 获取当前窗口大小
            QSize newSize = event->size();

            // 检查是否小于最小尺寸
            if (newSize.width() < minWidth || newSize.height() < minHeight)
            {
                // 调整窗口大小到最小值
                resize((std::max)(newSize.width(), minWidth), (std::max)(newSize.height(), minHeight));
                return;
            }

            // 刷新所有对象
            refresh_allObject();
        }

        void HalconWidget::mousePressEvent(QMouseEvent* event)
        {
            if (_isDrawingRect) {
                event->ignore(); // 如果正在绘制矩形，忽略鼠标事件
                return;
            }

            if (event->button() == Qt::LeftButton && rect().contains(event->pos())) {
                _isDragging = true;
                _lastMousePos = event->pos(); // 记录鼠标按下时的位置
                event->accept();
            }
            else {
                event->ignore();
            }
        }

        void HalconWidget::mouseMoveEvent(QMouseEvent* event)
        {
            if (_isDrawingRect) {
                event->ignore(); // 如果正在绘制矩形，忽略鼠标事件
                return;
            }

            if (_isDragging && _halconWindowHandle && !_halconObjects.empty()) {
                QPoint currentMousePos = event->pos();
                QPoint delta = currentMousePos - _lastMousePos; // 计算鼠标移动的偏移量

                // 获取当前显示区域
                HalconCpp::HTuple row1, col1, row2, col2;
                GetPart(*_halconWindowHandle, &row1, &col1, &row2, &col2);

                // 根据鼠标移动的偏移量调整显示区域
                double deltaX = -delta.x() * (col2.D() - col1.D() + 1) / width();
                double deltaY = -delta.y() * (row2.D() - row1.D() + 1) / height();

                double newCol1 = col1.D() + deltaX;
                double newCol2 = col2.D() + deltaX;
                double newRow1 = row1.D() + deltaY;
                double newRow2 = row2.D() + deltaY;

                // 设置新的显示区域
                SetPart(*_halconWindowHandle, newRow1, newCol1, newRow2, newCol2);

                // 清除窗口并重新显示所有对象
                ClearWindow(*_halconWindowHandle);
                for (auto& object : _halconObjects) {
                    if (object->isShow)
                    {
                        display_HalconWidgetDisObject(object);
                    }
                }

                _lastMousePos = currentMousePos; // 更新鼠标位置
                event->accept();
            }
            else {
                event->ignore();
            }
        }

        void HalconWidget::mouseReleaseEvent(QMouseEvent* event)
        {
            if (_isDrawingRect) {
                event->ignore(); // 如果正在绘制矩形，忽略鼠标事件
                return;
            }

            if (event->button() == Qt::LeftButton) {
                _isDragging = false; // 停止拖拽
                event->accept();
            }
            else {
                event->ignore();
            }
        }

        void HalconWidget::display_HalconWidgetDisObject(HalconWidgetDisObject* object)
        {
            if (object == nullptr || !object->has_value())
            {
                return; // 如果对象无效，直接返回
            }
            HalconCpp::HObject* halconObject = object->value();
            if (halconObject == nullptr || !halconObject->IsInitialized())
            {
                return; 
            }
            prepare_display(object->painterConfig);
            HalconCpp::HTuple hv_WindowHandle = *_halconWindowHandle;
			HalconCpp::DispObj(*halconObject, hv_WindowHandle);
        }

        void HalconWidget::prepare_display(const PainterConfig& config)
        {
            auto [r, g, b] = RQWColorToRGB(config.color);
            SetRgb(*_halconWindowHandle, r, g, b);
        }

        void HalconWidget::drawRect()
        {
            _isDrawingRect = true; // 开始绘制矩形
            HalconCpp::HTuple hv_Row1, hv_Column1, hv_Row2, hv_Column2;
            HalconCpp::HObject ho_Rectangle, ho_TemplateRegion, ho_MatchResult;

            // 调用 Halcon 的绘制矩形方法
            HalconCpp::DrawRectangle1(*_halconWindowHandle, &hv_Row1, &hv_Column1, &hv_Row2, &hv_Column2);

            // 生成矩形对象
            HalconCpp::GenRectangle1(&ho_Rectangle, hv_Row1, hv_Column1, hv_Row2, hv_Column2);

            // 提取矩形区域内的内容作为模板学习区域
            HalconCpp::ReduceDomain(*_halconObjects.front()->value(), ho_Rectangle, &ho_TemplateRegion);

            // 创建模板
            HalconCpp::HTuple hv_TemplateID, hv_HomMat2D;
            HalconCpp::CreateShapeModel(ho_TemplateRegion, "auto", -0.39, 0.79, "auto", "auto", "use_polarity", "auto", "auto", &hv_TemplateID);

            HalconCpp::HObject ho_ModelContours, ho_ContoursAffineTrans;
            HalconCpp::GetShapeModelContours(&ho_ModelContours, hv_TemplateID, 1);

            // 在整个图像中进行模板匹配
            HalconCpp::HTuple hv_Row, hv_Column, hv_Angle, hv_Score;
            HalconCpp::FindShapeModel(ho_TemplateRegion, hv_TemplateID, -0.39, 0.79, 0.5, 1, 0.5, "least_squares", 0, 0.9, &hv_Row, &hv_Column, &hv_Angle, &hv_Score);

            if ((hv_Row.TupleLength()) > 0)
            {
                // 计算仿射变换矩阵
                VectorAngleToRigid(0, 0, 0, hv_Row, hv_Column, hv_Angle, &hv_HomMat2D);

                // 将模板轮廓进行仿射变换
                HalconCpp::AffineTransContourXld(ho_ModelContours, &ho_ContoursAffineTrans, hv_HomMat2D);

                // 设置显示颜色
                HalconCpp::SetColor(*_halconWindowHandle, "blue");

                // 显示匹配到的轮廓
                HalconCpp::DispObj(ho_ContoursAffineTrans, *_halconWindowHandle);

                // 显示匹配分数
                for (int i = 0; i < hv_Score.TupleLength(); i++)
                {
                    HalconCpp::HTuple hv_ScoreText = std::to_string(hv_Score[i].D()).c_str() ;
                    HalconCpp::SetTposition(*_halconWindowHandle, hv_Row[i].D(), hv_Column[i].D());
                    HalconCpp::WriteString(*_halconWindowHandle, hv_ScoreText);
                }
            }
            else
            {
                // 如果没有匹配到，显示提示信息
                HalconCpp::SetColor(*_halconWindowHandle, "red");
                HalconCpp::SetTposition(*_halconWindowHandle, 10, 10);
                HalconCpp::WriteString(*_halconWindowHandle, "No match found!");
            }

            _isDrawingRect = false; // 绘制完成
        }

        HalconWidgetDisObject HalconWidget::drawRect(PainterConfig config)
        {
            prepare_display(config);
            _isDrawingRect = true;
            HalconCpp::HTuple hv_Row1, hv_Column1, hv_Row2, hv_Column2;

            auto ho_Rectangle = new HalconCpp::HObject;
            auto ho_Contour = new HalconCpp::HObject;

            // 绘制矩形
            HalconCpp::DrawRectangle1(*_halconWindowHandle, &hv_Row1, &hv_Column1, &hv_Row2, &hv_Column2);
            HalconCpp::GenRectangle1(ho_Rectangle, hv_Row1, hv_Column1, hv_Row2, hv_Column2);

            // 将矩形转换为轮廓（边框）
            HalconCpp::GenContourRegionXld(*ho_Rectangle, ho_Contour, "border");

            _isDrawingRect = false;

            // 创建 HalconWidgetDisObject 对象
            auto object = new HalconWidgetDisObject(ho_Contour); // 使用轮廓对象
            object->id = getVailidAppendId();
            object->painterConfig = config;
            object->isShow = true;
            object->type = HalconWidgetDisObject::ObjectType::Rectangle;
            object->descrption = "drawRect";

            appendHObject(object);
            return *object;
        }

        void HalconWidget::shapeModel(HalconWidgetDisObject& rec)
        {
            HalconCpp::HTuple hv_Row1, hv_Column1, hv_Row2, hv_Column2;
            HalconCpp::HObject ho_Rectangle, ho_TemplateRegion, ho_MatchResult;

            // 提取矩形区域内的内容作为模板学习区域
            HalconCpp::ReduceDomain(*_halconObjects.front()->value(), *rec._object, &ho_TemplateRegion);

            // 创建模板
            HalconCpp::HTuple hv_TemplateID, hv_HomMat2D;
            HalconCpp::CreateShapeModel(ho_TemplateRegion, "auto", -0.39, 0.79, "auto", "auto", "use_polarity", "auto", "auto", &hv_TemplateID);

            HalconCpp::HObject ho_ModelContours, ho_ContoursAffineTrans;
            HalconCpp::GetShapeModelContours(&ho_ModelContours, hv_TemplateID, 1);
        }

        void HalconWidget::study()
        {
        }
	}
}