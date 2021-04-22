#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: geofftong
# Mail: geofftong@tencent.com
# Created Time: 2019/11/15 6:22 PM

import json
import requests
import time
import jwt


URL = "https://openai.weixin.qq.com/openapi/nlp/news-abstraction/{TOKEN}"


def test():
    input_data = {
        "uid": "yourid",  # 能标识用户的唯一用户id，可以是openid
        "data": {
            "q": "恭喜小张脱单成功",
            "mode": "3class"}
    }

    # 调用jwt库,生成json web token
    jwt_token = jwt.encode(EncodingAESKey,  # payload, 有效载体
                           "yourkey",  # 进行加密签名的密钥
                           algorithm="HS256",  # 指明签名算法方式, 默认也是HS256
                           headers=headers  # json web token 数据结构包含两部分, payload(有效载体), headers(标头)
                           ).decode('ascii')  # python3 编码后得到 bytes, 再进行解码(指明解码的格式), 得到一个str

    print(jwt_token)


    category = '娱乐'
    title = '李念平即将离任中国驻哥伦比亚大使'
    content = '据中国驻哥伦比亚大使馆官网消息：7月11日，李念平大使辞行拜会哥伦比亚外长特鲁西略。特鲁西略外长对李大使离任表示惋惜，感谢李大使为推动哥中关系发展所作贡献。中国驻哥伦比亚大使馆官网图\\r\\n上述官方消息证实，资深外交官李念平即将离任中国驻哥伦比亚大使一职。官方简历显示，李念平（1963.7）先后担任过外交部西欧司副处长、驻德国大使馆参赞、外交部办公厅参赞兼处长、外交部欧洲司副司长等职。\\r\\2009年后，李念平历任驻德国大使馆公使衔参赞、公使等职，后于2013年转任宁夏回族自治区外事（侨务、港澳事务）办公室主任。李念平于2016年1月接替汪晓源出任中国驻哥伦比亚大使。1980年2月7日，中华人民共和国和哥伦比亚共和国建立外交关系。同年6月和9月，中哥互设大使馆。1989年11月，两国就互设领事馆达成协议，中国驻巴兰基亚领事馆于1990年6月开馆。2013年，哥在上海设立总领馆。同年，双方就哥在广州设立总领馆达成协议，10月30日广州总领馆正式开馆。'
    do_news_classify = True
    json_data = {'title': title, 'content': content, 'category': category, 'do_news_classify': do_news_classify}
    response = requests.post(url=URL, headers={'Content-Type': 'application/json'}, data=json.dumps(json_data))
    result_json = json.loads(response.text)
    for k, v in result_json.items():
        print("%s: %s" % (k, v))


if __name__ == '__main__':
    start = time.time()
    loop_num = 10
    for _ in range(loop_num):
        test()
    end = time.time()
    print("average time: %fs" % ((end - start) * 1.0 / loop_num))
