
#ifndef __SYS_CAPTURE_DUMP_H__
#define __SYS_CAPTURE_DUMP_H__

#include "SysObject.h"

// 记录线程崩溃时的堆栈
#define RECORD_DUMP(function)     int code = 0;\
    __try \
    { \
        function;\
    } \
    __except (code = GetExceptionCode(), SysDump::SaveDump(GetExceptionInformation()), EXCEPTION_EXECUTE_HANDLER) \
    { \
        MessageBoxW(NULL, L"程序出现异常！！！", L"提示", MB_OK); \
        if (isExitProcess) \
        { \
            ExitProcess(code); \
        }\
    }\
    return code

 /**
  * @brief 捕获线程异常，打印调用堆栈
  */
class COMMONLAYER_EXPORT SysCaptureDump : public SysObject
{
public:
    /**
     * @brief 析构函数
     */
    virtual ~SysCaptureDump();

    /**
     * @brief 捕获异常
     * @param isExitProcess 是否退出程序
     */
    int capture(bool isExitProcess = true);

protected:
    /**
     * @brief 构造函数
     * @declare 受保护权限保证外部不能实例化对象
     */
    SysCaptureDump();

    /**
     * @brief 禁用拷贝构造函数
     */
    SysCaptureDump(const SysCaptureDump&) = delete;

    /**
     * @brief 禁用赋值运算符
     */
    SysCaptureDump& operator=(const SysCaptureDump&) = delete;

    /**
     * @brief 线程运行函数
     * @declare 继承类在此函数中实现业务
     */
    virtual void onThreadRun() = 0;

};

#endif // __SYS_CAPTURE_DUMP_H__
