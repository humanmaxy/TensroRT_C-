#pragma once

#include "VB_global.h"
#include "SysCaptureDump.h"
#include <QtWidgets/QApplication>

 // 向前声明
class VBMainWindow;
class VBLogNotify;
class CatlThread;

class VB_EXPORT VBApplication : public QApplication, public SysCaptureDump
{
    Q_OBJECT

public:
    /**
     * @brief 构造函数
     */
    explicit VBApplication(int& argc, char** argv);

    /**
     * @brief 析构函数
     */
    virtual ~VBApplication();

signals:
    /**
     * @brief 通知鼠标按下的事件
     */
    void SigMouseButtonPress();

    /**
     * @brief 通知加载自定义工具问题信息
     */
    void SigDynamicToolsLoadErr(const QString& errMsg);

    void SigVisionProNotInstall(const QString& errMsg);

    void SigVisionproCallback(std::string toolId, bool ioChanged);

protected:
    /**
     * @brief 事件通知
     */
    bool notify(QObject* obj, QEvent* event) Q_DECL_OVERRIDE;

    /**
     * @brief 运行函数
     */
    void onThreadRun() override;

    /**
     * @brief 初始化界面
     */
    void Initialise();

    /**
     * @brief 初始化日志
     */
    void InitialiseLog();
    
    /**
     * @brief 初始化Dump路径
     */
    void InitialiseDump();

    /**
     * @brief 初始化全局产品参数路径
     */
    void InitialiseGlobalParam();

    /**
     * @brief 初始化工具类
     */
    void InitialiseTool(QString& dynamicToolsLoadErr, const QString& path);

    /**
     * @brief 初始化单例
     */
    void InitialiseInstances();

    /**
     * @brief 加载翻译文件
     */
    void LoadTranslation();

    /**
     * @brief 初始化合成、组合工具、for循环工具
     */
    void LoadTools();

    /**
     * @brief 初始化合成工具
     */
    void LoadBranchTools(const std::string& strSaveFloder);

    /**
     * @brief 初始化组合工具
     */
    void LoadComposeTools(const std::string& strSaveFloder);

    /**
     * @brief 初始化VisonPro工具
     */
    void LoadVisionProTools(const std::string& strSaveFloder);

    /**
     * @brief 初始化for循环工具
     */
    void LoadCirculateTools(const std::string& strSaveFloder);

    /**
      * @brief 加载锁信息
      */
    void LoadMutexInfos(const std::string& strSaveFloder);

    /**
     * @brief 释放资源
     */
    void UnInitialise(int iRel);

    /**
     * @brief 析构单例
     */
    void UnInitialiseInstances();

    /**
     * @brief 释放工具类
     */
    void UnInitialiseTool();

    /**
     * @brief 释放日志
     */
    void UnInitialiseLog();

private:
    // 主体窗口
    VBMainWindow* m_pVBMainWindow;
    VBLogNotify* m_pNofity;
    CatlThread* m_pThread;
    bool m_bLoadLibrary;
    bool m_bLoadVisionProLibrary;

};
