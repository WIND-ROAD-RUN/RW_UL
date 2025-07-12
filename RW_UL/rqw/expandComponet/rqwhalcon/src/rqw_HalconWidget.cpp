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
            name(other.name),
            isShow(other.isShow),
			type(other.type)
        {
        }

		HalconWidgetDisObject::HalconWidgetDisObject(HalconWidgetDisObject&& other) noexcept
            : _object(other._object),
            id(other.id),
            name(std::move(other.name)),
            isShow(other.isShow),
			type(other.type)
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
                name = other.name;
                isShow = other.isShow;
				type = other.type;
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
                name = std::move(other.name);
                isShow = other.isShow;
				type = other.type;

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

        HalconWidgetDisObjectId HalconWidget::getMinValidAppendId()
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
                    DispObj(*object->_object, *_halconWindowHandle);
                }
            }
        }

        void HalconWidget::updateWidget()
        {
            refresh_allObject();
        }


        void HalconWidget::appendVerticalLine(int position, const HalconWidgetDisObjectPainterConfig& config)
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
            SetColor(*_halconWindowHandle, config.color == HalconWidgetDisObjectPainterConfig::Color::Black ? "black" : "white");

            // 创建并添加对象
            HalconWidgetDisObject object(verticalLine);
            object.isShow = true;
            object.painterConfig = config;
            object.type = HalconWidgetDisObject::ObjectType::Line;
            object.id = getMinValidAppendId();
            appendHObject(object);
        }

        void HalconWidget::appendHorizontalLine(int position, const HalconWidgetDisObjectPainterConfig& config)
        {
            if (!_halconWindowHandle)
            {
                return;
            }

            // 获取窗口的高度和宽度范围
            HalconCpp::HTuple row1, col1, row2, col2;
            GetPart(*_halconWindowHandle, &row1, &col1, &row2, &col2);

            // 校准 position，使其基于窗口的坐标范围
            int calibratedPosition = static_cast<int>(row1.D() + position);

            // 检查校准后的 position 是否在有效范围内
            if (calibratedPosition - config.thickness / 2 < row1.D() || calibratedPosition + config.thickness / 2 > row2.D())
            {
                throw std::out_of_range("Position is out of the valid range for the horizontal line.");
            }

            // 生成水平线
            auto horizontalLine = new HalconCpp::HObject;
            HalconCpp::GenRectangle1(horizontalLine, calibratedPosition - config.thickness / 2, col1.D(), calibratedPosition + config.thickness / 2, col2.D());

            // 设置颜色
            SetColor(*_halconWindowHandle, config.color == HalconWidgetDisObjectPainterConfig::Color::Black ? "black" : "white");

            // 创建并添加对象
            HalconWidgetDisObject object(horizontalLine);
            object.isShow = true;
            object.painterConfig = config;
            object.type = HalconWidgetDisObject::ObjectType::Line;
            object.id = getMinValidAppendId();
            appendHObject(object);
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

            if (rect().contains(event->position().toPoint())) { // 检查鼠标是否在 HalconWidget 内
                int delta = event->angleDelta().y(); // 获取滚轮滚动的角度
                double scaleFactor = (delta > 0) ? 1.1 : 0.9; // 缩放因子，向上滚动放大，向下滚动缩小

                // 获取鼠标在 HalconWidget 中的位置
                QPointF mousePos = event->position();
                int mouseX = static_cast<int>(mousePos.x());
                int mouseY = static_cast<int>(mousePos.y());

                // 获取当前显示区域
                HalconCpp::HTuple row1, col1, row2, col2;
                GetPart(*_halconWindowHandle, &row1, &col1, &row2, &col2);

                // 计算当前显示区域的宽高
                double currentWidth = col2.D() - col1.D() + 1;
                double currentHeight = row2.D() - row1.D() + 1;

                // 计算鼠标位置在图像中的相对位置
                double relativeX = col1.D() + (mouseX / static_cast<double>(width())) * currentWidth;
                double relativeY = row1.D() + (mouseY / static_cast<double>(height())) * currentHeight;

                // 计算新的显示区域
                double newWidth = currentWidth / scaleFactor;
                double newHeight = currentHeight / scaleFactor;
                double newCol1 = relativeX - (mouseX / static_cast<double>(width())) * newWidth;
                double newRow1 = relativeY - (mouseY / static_cast<double>(height())) * newHeight;
                double newCol2 = newCol1 + newWidth - 1;
                double newRow2 = newRow1 + newHeight - 1;

                // 清除窗口内容
                ClearWindow(*_halconWindowHandle);

                // 设置新的显示区域
                SetPart(*_halconWindowHandle, newRow1, newCol1, newRow2, newCol2);

                // 重新显示所有对象
                for (auto& object : _halconObjects)
                {
                    if (object->isShow)
                    {
                        DispObj(*object->_object, *_halconWindowHandle);
                    }
                }

                event->accept(); // 事件已处理
            }
            else {
                event->ignore(); // 事件未处理
            }
        }

        void HalconWidget::showEvent(QShowEvent* event)
        {
            refresh_allObject();
        }

        void HalconWidget::resizeEvent(QResizeEvent* event)
        {
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
                        DispObj(*object->_object, *_halconWindowHandle);
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

        void HalconWidget::drawRect()
        {
            _isDrawingRect = true; // 开始绘制矩形
            HalconCpp::HTuple hv_Row1, hv_Column1, hv_Row2, hv_Column2;
            HalconCpp::HObject ho_Rectangle;

            // 调用 Halcon 的绘制矩形方法
            HalconCpp::DrawRectangle1(*_halconWindowHandle, &hv_Row1, &hv_Column1, &hv_Row2, &hv_Column2);

            // 生成矩形对象并显示
            HalconCpp::GenRectangle1(&ho_Rectangle, hv_Row1, hv_Column1, hv_Row2, hv_Column2);
            //appendHObject(ho_Rectangle);
        	HalconCpp::DispObj(ho_Rectangle, *_halconWindowHandle);

            _isDrawingRect = false; // 绘制完成
        }
	}
}