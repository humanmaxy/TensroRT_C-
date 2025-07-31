#ifndef __SYS_OBJECT_H__
#define __SYS_OBJECT_H__

#include <map>
#include <string>
#include "CommonGlobal.h"

// 向前声明
class SysObjectPrivate;

class COMMONLAYER_EXPORT SysObject
{
public:
    /**
     * @brief 析构函数
     */
    virtual ~SysObject();

protected:
    /**
     * @brief 构造函数
     * @declare 受保护权限保证外部不能实例化对象
     */
    SysObject();

    /**
     * @brief 禁用拷贝构造函数
     */
    SysObject(const SysObject&) = delete;

    /**
     * @brief 禁用赋值运算符
     */
    SysObject& operator=(const SysObject&) = delete;

protected:
    /**
     * @brief 获取对象的私有类
     * @param str[in] 字符串
     */
    SysObjectPrivate* GetObjectPrivate(const std::string& str) const;

    /**
     * @brief 设置对象的私有类
     * @param str[in]  字符串
     *        pObj[in] 私有类
     */
    void SetObjectPrivate(const std::string& str, SysObjectPrivate* pObj);

private:
    // 定义数据类型
    using ObjPrivateMap = std::map<std::string, SysObjectPrivate *>;

    // 导出类对应的私有实现类
    ObjPrivateMap m_mapObjPrivate;

};

// 设置/获取对象的私有类
#define SetPrivate(x)           SetObjectPrivate(CLASS_NAME(x), new x())
#define GetPrivate(x)           dynamic_cast<x *>(GetObjectPrivate(CLASS_NAME(x)))
#define GetObjPrivate(obj, x)   dynamic_cast<x *>(obj.GetObjectPrivate(CLASS_NAME(x)))

#endif // __SYS_OBJECT_H__
