# 开放接口文档


## 配置信息

例如:

APPID: xxxxxxxxxxxxxxx

TOKEN: xxxxxxxxxxxxxxx

EncodingAESKey: xxxxxxxxxxxxxxx

## 接口信息

### 新闻摘要接口(只签名不加密):

`https://openai.weixin.qq.com/openapi/nlp/news-abstraction/{{TOKEN}`

接口类型:

`POST请求`


### 参数说明:

字段|类型|默认值|描述
--|--|--|--
query|string||使用JWT签名后的数据

query签名说明:

字段|类型|默认值|描述
--|--|--|--
uid|string|自动生成的随机标识|用户标识的唯一ID，比如：openid
data|json Object||详见下面的data说明

data说明:

| **参数**  |                  **类型**                   | **说明** |
| :-------: | :-------: | :--------------------------------------: |
|    title    | string | 新闻标题(UTF-8)，可为空|
| content | string | 新闻正文内容(UTF-8)，不同段落间分隔符为'\\\r\\\n' |
| category | string |（可选）新闻所属的一级类别标签，如“时事”等 |
| do_news_classify | bool |（可选）是否在提取摘要前先进行新闻分类标志 |

注：对于参数do_news_classify，如标志为True则先进行新闻分类，若分类结果为"不适合"则跳过后续的摘要提取，如标志为False则只进行新闻摘要提取。

使用[JSON Web Token](https://www.jsonwebtoken.io/)的 `HS256` 算法对参数进行encode, 放入到query参数中

使用 jwt 和 `EncodingAESKey` 对数据对象进行encode得到加密字符串

```js
const signedData = jwths256.encode(EncodingAESKey, {
      uid: "xjlsj33lasfaf", //能标识用户的唯一用户id，可以是openid
      data: {
        "title": "大标题: 周杰伦透露孩子性格像自己",
        "content" "自侃：在家地位不高周杰伦晒与儿子\\r\\n周杰伦与妻子昆凌\\r\\n新浪娱乐讯据台湾媒体报道，周杰伦今受邀出席公益篮球活动，许久没公开亮相的他坦言：“大家看我的社交网站就知道，现在玩乐和陪家人是最重要的。”他预计10月将启动全新巡演，但聊到新专辑，他笑说：“目前已经做好5首，但包括上次做好的2首。”进度严重落后，巡演前能否来得及推出仍是未知数。\r\\n周杰伦曾被“老萧”萧敬腾与五月天阿信调侃：“巴黎铁塔好像是他家一样”，周杰伦也坦言有考虑在法国置产，地点属意：“有巴黎铁塔蛮重要的”，他感叹身处欧洲比较可以自由自在走在街上，就算被粉丝认出，打个招呼就滑着滑板溜走，不用跑得太狼狈。\\r\\n今天他与小朋友共享篮球时光，但聊到自己的一双儿女，他说：“小朋友个性像自己，比较顽固一点，小朋友都是这样、比较讲不听，严格来说我扮黑脸也是没什么用，在家里地位不是很高。”他形容女儿就像自己“另个女朋友”，有时候想和她和好还被拒绝，一直被闹别扭，周杰伦无奈说：“我还不知道怎么对待一个女儿。”倒是儿子比较怕他，只要一出声就会低头认错，“像他会画在桌子上，家里很多画，就会教他看画手要放在后面，不要摸”严父模样只有儿子会买单。\\r\\n阿信曾夸周杰伦是“华人音乐圈精神领袖”，周杰伦赞阿信是“心目中真正的音乐人”，曾邀阿信到家中作客，2人畅谈音乐，阿信还精辟分析他专辑，“发现他是实际有在听我歌曲的人”，但问到每次打球都只邀萧敬腾，周杰伦笑说有透过社交网站约阿信打球，每次却只被回表情符号，忍不住说：“他说他以前打曲棍球，我是不太相信的。”\\r\\n周杰伦将于10月启动全新巡演，谈到近期的困扰，就是还没想到一个很威的名字，“想不到比‘地表最强’更好的，但这次跟（出道）20年有关，会比较欢乐，不是走自己多强势的感觉”。\\r\\n(责编：漠er凡)\\r\\n":
        "category": "娱乐"
        "do_news_classify": true
      }
    }
)
```

### 调用开放平台语义接口

```js
curl -XPOST -d "query=signedData" https://openai.weixin.qq.com/openapi/nlp/news-abstraction/{TOKEN}
```

<a href="https://www.jsonwebtoken.io/" target="_blank">https://www.jsonwebtoken.io/</a>

> Tips: 在 [jsonwebtoken.io](https://www.jsonwebtoken.io) 网站上可以参考如下步骤手动生成signedData
![手动生成signedData示例](../assets/jwt.png)


### 返回结果

```json
{
  "abstraction": 新浪娱乐讯据台湾媒体报道，周杰伦今受邀出席公益篮球活动，许久没公开亮相的他坦言：“大家看我的社交网站就知道，现在玩乐和陪家人是最重要的。”他预计10月将启动全新巡演，但聊到新专辑，他笑说：“目前已经做好5首，但包括上次做好的2首。”进度严重落后，巡演前能否来得及推出仍是未知数。
  "classification": true,
  "prob": 0.999969363213,
}
```

#### 返回结果说明

| **字段**  |                  **类型**                  | **说明** |
| :-------: | :--------------------------------------: | :-------: |
|    abstraction     | string | 摘要提取的结果 |
|    classification     | bool | 是否适合提取摘要的分类标签, 如输入的do_news_classify为False，则默认返回为true |
|    prob     | float | 新闻分类结果对应的得分，如输入的do_news_classify为False，则默认返回1.0 |
