#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: geofftong
# Mail: geofftong@tencent.com
# Created Time: 2019/11/13 3:10 PM
import json

import requests
import time

URL = "http://9.141.138.122/news-abstraction"


def test():
    category = '娱乐'
    title = '大标题: 周杰伦透露孩子性格像自己'
    content = '自侃：在家地位不高周杰伦晒与儿子\\r\\n周杰伦与妻子昆凌\\r\\n新浪娱乐讯据台湾媒体报道，周杰伦今受邀出席公益篮球活动，许久没公开亮相的他坦言：“大家看我的社交网站就知道，现在玩乐和陪家人是最重要的。”他预计10月将启动全新巡演，但聊到新专辑，他笑说：“目前已经做好5首，但包括上次做好的2首。”进度严重落后，巡演前能否来得及推出仍是未知数。\r\\n周杰伦曾被“老萧”萧敬腾与五月天阿信调侃：“巴黎铁塔好像是他家一样”，周杰伦也坦言有考虑在法国置产，地点属意：“有巴黎铁塔蛮重要的”，他感叹身处欧洲比较可以自由自在走在街上，就算被粉丝认出，打个招呼就滑着滑板溜走，不用跑得太狼狈。\\r\\n今天他与小朋友共享篮球时光，但聊到自己的一双儿女，他说：“小朋友个性像自己，比较顽固一点，小朋友都是这样、比较讲不听，严格来说我扮黑脸也是没什么用，在家里地位不是很高。”他形容女儿就像自己“另个女朋友”，有时候想和她和好还被拒绝，一直被闹别扭，周杰伦无奈说：“我还不知道怎么对待一个女儿。”倒是儿子比较怕他，只要一出声就会低头认错，“像他会画在桌子上，家里很多画，就会教他看画手要放在后面，不要摸”严父模样只有儿子会买单。\\r\\n阿信曾夸周杰伦是“华人音乐圈精神领袖”，周杰伦赞阿信是“心目中真正的音乐人”，曾邀阿信到家中作客，2人畅谈音乐，阿信还精辟分析他专辑，“发现他是实际有在听我歌曲的人”，但问到每次打球都只邀萧敬腾，周杰伦笑说有透过社交网站约阿信打球，每次却只被回表情符号，忍不住说：“他说他以前打曲棍球，我是不太相信的。”\\r\\n周杰伦将于10月启动全新巡演，谈到近期的困扰，就是还没想到一个很威的名字，“想不到比‘地表最强’更好的，但这次跟（出道）20年有关，会比较欢乐，不是走自己多强势的感觉”。\\r\\n(责编：漠er凡)\\r\\n'
    # category = ''
    # title = '董卿、康辉、撒贝宁聚集，我却爱上了这个女人！'
    # content = '作者｜柚子\\r\\n最近央视的《主持人大赛》火了。\\r\\n台上各路选手精彩过招，那场面堪比“神仙打架”。\\r\\n台下的嘉宾们，也看点颇多。\\r\\n身为点评嘉宾的董卿和康辉，点评一针见血，仅仅听着就是一种享受。\\r\\n但最让我惊喜的，还是专业评委席上一个久违的面孔――敬一丹。\\r\\n这位主持了《焦点访谈》20多年的女主播，今年64岁。\\r\\n4年前，她从央视退休，如今，又回到央视做评委，也算是另一种意义上的“轮回”吧。\\r\\n虽然节目中，敬一丹的镜头不多，却让人无法忽视她的存在。\\r\\n每当有选手上台，她都会赞许地点点头，你能从她的眼神里，感受到鼓舞。\\r\\n还有就是每次镜头扫到她，敬一丹都会微微一笑，一瞬间，让人如沐春风。\\r\\n温柔，又有力量。\\r\\n（图源：《中央广播电视总台2019主持人大赛》）\\r\\n这种骨子里散发出的温柔，谁能抵挡？\\r\\n难怪有人说，“主持人大赛，只看敬一丹老师打分就可以了。”\\r\\n其实不止性格温柔，敬一丹的人生，也是柔和的。\\r\\n如果用四个字来概括，那就是――\\r\\n不急不躁。\\r\\n01\\r\\n28岁，考上研究生\\r\\n张爱玲说，出名要趁早。\\r\\n不同于其他央视名嘴“从广院毕业，直接进入央视”这一气呵成的人生，敬一丹的成名之路，是有些曲折的。\\r\\n17岁那年，她中学毕业，在一个林区广播站做播音员，和广播结下了不解之缘。\\r\\n（图源：《鲁豫有约》之《敬一丹・耳顺之年，退休之后》）\\r\\n之后，她又被推荐到广院（中传前身）学习。\\r\\n毕业后成为黑龙江人民广播电视台的一名主持人。\\r\\n（图源：《鲁豫有约》之《敬一丹・耳顺之年，退休之后》）\\r\\n按理说，做着一份自己喜欢并体面的工作，人生已经算成功了。\\r\\n可敬一丹，觉得自己不止于此。\\r\\n她想接受更系统的教育，学习更多专业知识。\\r\\n于是在25岁那年，她一边工作，一边准备考研。\\r\\n因为大学时没开外语课，考研必考科目英语，就够让她头疼的。\\r\\n（图源：《鲁豫有约》之《敬一丹・耳顺之年，退休之后》）\\r\\n那时候，她连最基本的26个字母都认不全，一切都要从零开始。\\r\\n敬一丹说，她几乎把所有的业余时间，都用在了考研上。\\r\\n可第一年，还是意料之中落榜了。\\r\\n敬一丹并没有气馁，她心里只有一个想法：我要继续考。\\r\\n谁知第二年，她又落榜了。\\r\\n其实当时的敬一丹，已经不想再折腾了。她觉得自己底子差，和其他人差距也大，继续考下去，真的有必要吗？\\r\\n但这条路走了两年，说放弃，也不甘心，于是她咬紧牙关，又考了一次。\\r\\n第三次，敬一丹终于如愿以偿，考上了北广的研究生。\\r\\n（图源：《鲁豫有约》之《敬一丹・耳顺之年，退休之后》）\\r\\n有时候放弃很容易，坚持下去，却很难。\\r\\n回头看，黎明前的黑暗，也很美。\\r\\n02\\r\\n33岁，被央视录取\\r\\n不知你是否也听过这样一句话，“你都30多岁了，别再瞎折腾了。”\\r\\n30岁左右，似乎是人生的分水岭。\\r\\n站在人生的分叉路口，是继续向前走，还是回归平淡？在敬一丹这里，也有过纠结。\\r\\n敬一丹在读完研究生后，留在北广当老师，一当就是三年。\\r\\n稳定的工作，不错的收入，任谁看，都是一个肥差。\\r\\n可敬一丹，又不太想安于现状了。\\r\\n（图片来源：网络）\\r\\n她觉得，自己既然是学新闻的，就应该干一些更有挑战性的事。\\r\\n当时正好央视来北广招人，敬一丹就想去试试。\\r\\n可当她把这个想法告诉周围人时，几乎没人支持她。\\r\\n理由无外乎：你都30多岁了。\\r\\n可最后，敬一丹还是遵循自己的内心，去参加了面试。\\r\\n（图源：网络）\\r\\n她说，“如果我听从他们的意见，也许我会一辈子在广院当老师，永远过着波澜不惊的生活，那将是我一辈子的遗憾。”\\r\\n是啊，有些事，做了，失败会后悔一时，但不做，却要后悔一辈子。\\r\\n如果把人生每一个选择比作是一场博弈，那这一次，敬一丹赌赢了。\\r\\n33岁的她如愿进入央视。\\r\\n但压力，也随之而来。\\r\\n不断有新鲜血液涌入台里，敬一丹的年龄，是有些尴尬的。\\r\\n为了追赶上年轻人的步伐，她只能私下付出更多努力。她也会在遇到问题时，谦虚地去请教小辈。\\r\\n（图源：网络）\\r\\n在新闻事业上获得的成就感，是敬一丹当老师时不曾有过的。\\r\\n总有人说，你在什么年纪，就应该做什么年纪该做的事。\\r\\n20岁左右，就应该去折腾，30岁左右，就应该结婚生子，40岁左右，一切安定下来。\\r\\n好像一旦偏离这个轨迹，就违背了“人生规律”。\\r\\n可做自己喜欢的事，哪有什么早晚？\\r\\n03\\r\\n40岁，迎来事业黄金期\\r\\n人生选择上不紧不慢外，敬一丹身上的另一种柔和，来自对年龄的从容。\\r\\n敬一丹说过两个故事。\\r\\n一个是她40岁那年去学车，一开始总出错，年轻的教练就问她，“你今年多大了？”\\r\\n敬一丹说，“40啊。”\\r\\n教练听了不可思议地说，“这么大岁数，找这个刺激干嘛？”\\r\\n可敬一丹不以为意，她心想，“40岁，算个事儿吗？”\\r\\n（图片来源：北京卫视《人生相对论》）\\r\\n在她这里，还真不算。不仅如此，她还在40岁这年，接受了更大的挑战。\\r\\n那一年，央视要办《焦点访谈》，这个节目的分量有多重呢？\\r\\n它是一档前所未有的舆论监督节目，还在《新闻联播》后的黄金时间播出。\\r\\n当敬一丹知道自己要担任这档节目的主持人时，她的眼睛都亮了。\\r\\n（图片来源：北京卫视《人生相对论》）\\r\\n40岁，或许对很多人来说，早过了事业发力期，精力和体力都有些跟不上了。\\r\\n可敬一丹，毫不犹豫地接过这个重担。\\r\\n对她来说，这既然是自己想做的事情，就不会感到疲惫。\\r\\n后来的故事我们都知道了，《焦点访谈》迅速成为家喻户晓的节目，敬一丹也在不惑之年，迎来了事业巅峰期。\\r\\n人生不可逆，但灵魂，却可以永远年轻，不是吗？\\r\\n（图片来源：北京卫视《人生相对论》）\\r\\n敬一丹对年龄的从容，也体现在对外表衰老这件事的理解上。\\r\\n有一次，鲁豫问60岁的敬一丹，“你介意别人提你的年龄吗？”\\r\\n敬一丹说――\\r\\n“我在哪个年龄段，就会觉得这个年龄真好。我不会在40岁的时候，觉得20岁真好，那么到了50岁的时候，我觉得50岁真好，此刻我觉得60岁真好。”\\r\\n（图源：《鲁豫有约》之《敬一丹・耳顺之年，退休之后》）\\r\\n不知从什么时候起，越来越多人被“少女感”裹挟，坦率接受变老这件事的人却很少。\\r\\n身边很多女性，会因为年龄的增长陷入焦虑，越来越松弛的皮肤，越来越多的皱纹.......\\r\\n可大家往往忽视了一点――\\r\\n虽然青春不再，但30岁，40岁，50岁的丰富阅历和内心的从容淡定，是20岁不会拥有的。这才是最宝贵的人生财富。\\r\\n而敬一丹，深谙这个道理。\\r\\n04\\r\\n60岁，是崭新的20岁\\r\\n很多人在退休之后，都会有一种落寞感――\\r\\n感觉自己不再被需要了，感觉剩下的日子，就得过且过地过着吧。\\r\\n可敬一丹，却在60岁那年，做了自己以后20多年的人生规划。\\r\\n（图片来源：网络）\\r\\n听起来很不可思议对吧？\\r\\n其实，这是敬一丹从一个90多岁的老人那里得到的启发。\\r\\n她认识老人的时候，对方已经95岁了。\\r\\n老人就是在60岁那年，给自己定下了未来30年的计划。\\r\\n等到她90岁的时候，她完成了自己的所有计划，开办了一份老年社区报纸，写了几本书。\\r\\n（图片来源：北京卫视《人生相对论》）\\r\\n看了这位老人，敬一丹内心更加笃定：第二个青春开始了。\\r\\n（图片来源：北京卫视《人生相对论》）\\r\\n退休，便意味着崭新的开始。\\r\\n忙活了大半辈子，她终于有时间，去弥补自己的遗憾了。\\r\\n她先去了南极，后来又去了北极。她说，这是给自己的退休礼物。\\r\\n（图源：《鲁豫有约》之《敬一丹・耳顺之年，退休之后》）\\r\\n为什么要去那么远的地方呢？\\r\\n那是因为除了《焦点访谈》，敬一丹一直想做一个人文地理节目，她觉得可以全世界到处走，很酷。\\r\\n可到退休，也没找到合适的机会。\\r\\n敬一丹说，趁着身体还算硬朗，先去比较远的地方转转，等年纪再大一些，就到近一点的地方转转。\\r\\n从远及近，慢慢来。\\r\\n（图源：《鲁豫有约》之《敬一丹・耳顺之年，退休之后》）\\r\\n除了旅行，她还和女儿一起参加公益事业，用另一种方式继续探索这个世界。\\r\\n对敬一丹来说，60岁算什么？人生才刚刚开始。\\r\\n05\\r\\n每个人都有属于自己的时区\\r\\n25岁考研，28岁才考上。33岁，“高龄”进入央视。40岁，才开始做《焦点访谈》。60岁，开启一段崭新的人生。\\r\\n敬一丹的人生经历，让我想起之前刷爆朋友圈的一段演讲：\\r\\n“有人22岁就毕业了，但等了五年才找到好工作！有人25岁就当上了CEO，却在50岁去世了。也有人直到50岁才当上CEO，最后活到90岁。\\r\\n有人依然单身，而别人却早已结婚。奥巴马55岁退任总统，而川普却是70岁才开始当。\\r\\n世上每个人都有自己的发展时区。身边有些人看似走在你前面，也有人看似走在你后面。\\r\\n但其实每个人在自己的时区有自己的步程。不用嫉妒或嘲笑他们。他们都在自己的时区，你在你的！”\\r\\n（图源：《鲁豫有约》之《敬一丹・耳顺之年，退休之后》）\\r\\n之前鲁豫采访敬一丹的时候问她，“你崩溃过吗？”\\r\\n敬一丹说，“几乎没有。”\\r\\n我想，不被周围的人和事左右，在自己的时区里，顺其自然地走着，就是她一生平和的主要原因吧。\\r\\n正如她在自传里写的一句话，“迟钝，也许成全了我。”\\r\\n参考资料：\\r\\n敬一丹自传《我遇到你》\\r\\n北京卫视《人生相对论》\\r\\n《鲁豫有约》之《敬一丹・耳顺之年，退休之后》'
    do_news_classify = True
    json_data = {'title': title, 'content': content, 'category': category, 'do_news_classify': do_news_classify}
    response = requests.post(url=URL, headers={'Content-Type': 'application/json'}, data=json.dumps(json_data))
    result_json = json.loads(response.text)
    for k, v in result_json.items():
        print("%s: %s" % (k, v))


if __name__ == '__main__':
    start = time.time()
    loop_num = 3
    for _ in range(loop_num):
        test()
    end = time.time()
    print("average time: %fs" % ((end - start) * 1.0 / loop_num))