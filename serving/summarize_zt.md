# 功能介绍

本服务目前支持对输入新闻进行是否适合提取摘要的分类， 同时支持对给定新闻进行摘要自动提取。


# API文档

## 接口信息

- HTTP方法： `POST`
- 请求URL： `http://9.141.138.122/news-abstraction`
- 请求参数：

| **参数**  |                  **类型**                   | **说明** |
| :-------: | :-------: | :--------------------------------------: |
|    title    | string | 新闻标题(UTF-8)，可为空|
| content | string | 新闻正文内容(UTF-8)，不同段落间分隔符为'\\\r\\\n' |
| category | string |（可选）新闻所属的一级类别标签，如“时事”等 |
| do_news_classify | bool |（可选）是否在提取摘要前先进行新闻分类标志 |

注：对于参数do_news_classify，如标志为True则先进行新闻分类，若分类结果为"不适合"则跳过后续的摘要提取，如标志为False则只进行新闻摘要提取。

### 返回结果

| **字段**  |                  **类型**                  | **说明** |
| :-------: | :--------------------------------------: | :-------: |
|    abstraction     | string | 摘要提取的结果 |
|    classification     | string | 是否适合提取摘要的分类标签, 如输入的do_news_classify为False，则默认返回为"适合" |
|    prob     | float | 新闻分类结果对应的得分，如输入的do_news_classify为False，则默认返回1.0 |

说明：request和response都是json格式。