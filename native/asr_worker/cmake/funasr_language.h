#pragma once

#include <cstring>

namespace puripuly {

inline const char * funasr_language_name(const char * code) {
    static constexpr struct {
        const char * code;
        const char * name;
    } languages[] = {
        {"zh", "中文"},
        {"en", "英文"},
        {"yue", "粤语"},
        {"ja", "日文"},
        {"ko", "韩文"},
        {"vi", "越南语"},
        {"id", "印尼语"},
        {"th", "泰语"},
        {"ms", "马来语"},
        {"tl", "菲律宾语"},
        {"ar", "阿拉伯语"},
        {"hi", "印地语"},
        {"bg", "保加利亚语"},
        {"hr", "克罗地亚语"},
        {"cs", "捷克语"},
        {"da", "丹麦语"},
        {"nl", "荷兰语"},
        {"et", "爱沙尼亚语"},
        {"fi", "芬兰语"},
        {"el", "希腊语"},
        {"hu", "匈牙利语"},
        {"ga", "爱尔兰语"},
        {"lv", "拉脱维亚语"},
        {"lt", "立陶宛语"},
        {"mt", "马耳他语"},
        {"pl", "波兰语"},
        {"pt", "葡萄牙语"},
        {"ro", "罗马尼亚语"},
        {"sk", "斯洛伐克语"},
        {"sl", "斯洛文尼亚语"},
        {"sv", "瑞典语"},
    };
    for (const auto & language : languages) {
        if (std::strcmp(code, language.code) == 0) {
            return language.name;
        }
    }
    return code;
}

}
