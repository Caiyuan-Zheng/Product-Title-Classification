'''
爬虫介绍：
本程序用于在京东上爬取商品标题，爬取的方式有通过每个类别的链接和通过关键字两种方式。
程序存在的问题有：
(1)无法进行多线程爬虫
(2)由于网页格式的原因，有些类别的商品无法解析出来
'''
import requests
from lxml import etree
import time
import csv
import threading
import pickle
#定义函数抓取每页前30条商品信息


'''
参数
fout:输出到文件
n:第几页
keyword:类别
href:网址
lastPages:上一页的内容，防止重复

'''
def crow_first(fout,n,keyword,href,lastPages,current_num):
    #构造每一页的url变化
    url=None
    if n==1:
        url=href
    else:
        url=href+'&page='+str(n)+'&sort=sort_totalsales15_desc&trans=1&JL=6_0_0#J_mai'
    #url=href
    #url='https://search.jd.com/Search?keyword='+keyword.split('--')[2]+'&enc=utf-8&qrst=1&rt=1&stop=1&vt=2&wq='+keyword.split('--')[2]+'page='+str(2*n-1)+'&s=356&click=0'
    head = {'authority': 'search.jd.com',
            'method': 'GET',
            'path': '/s_new.php?keyword=%E6%89%8B%E6%9C%BA&enc=utf-8&qrst=1&rt=1&stop=1&vt=2&wq=%E6%89%8B%E6%9C%BA&cid2=653&cid3=655&page=4&s=84&scrolling=y&log_id=1529828108.22071&tpl=3_M&show_items=7651927,7367120,7056868,7419252,6001239,5934182,4554969,3893501,7421462,6577495,26480543553,7345757,4483120,6176077,6932795,7336429,5963066,5283387,25722468892,7425622,4768461',
            'scheme': 'https',
            'referer': 'https://search.jd.com/Search?keyword=%E6%89%8B%E6%9C%BA&enc=utf-8&qrst=1&rt=1&stop=1&vt=2&wq=%E6%89%8B%E6%9C%BA&cid2=653&cid3=655&page=3&s=58&click=0',
            'user-agent': 'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/66.0.3359.139 Safari/537.36',
            'x-requested-with': 'XMLHttpRequest',
            'Cookie':'qrsc=3; pinId=RAGa4xMoVrs; xtest=1210.cf6b6759; ipLocation=%u5E7F%u4E1C; _jrda=5; TrackID=1aUdbc9HHS2MdEzabuYEyED1iDJaLWwBAfGBfyIHJZCLWKfWaB_KHKIMX9Vj9_2wUakxuSLAO9AFtB2U0SsAD-mXIh5rIfuDiSHSNhZcsJvg; shshshfpa=17943c91-d534-104f-a035-6e1719740bb6-1525571955; shshshfpb=2f200f7c5265e4af999b95b20d90e6618559f7251020a80ea1aee61500; cn=0; 3AB9D23F7A4B3C9B=QFOFIDQSIC7TZDQ7U4RPNYNFQN7S26SFCQQGTC3YU5UZQJZUBNPEXMX7O3R7SIRBTTJ72AXC4S3IJ46ESBLTNHD37U; ipLoc-djd=19-1607-3638-3638.608841570; __jdu=930036140; user-key=31a7628c-a9b2-44b0-8147-f10a9e597d6f; areaId=19; __jdv=122270672|direct|-|none|-|1529893590075; PCSYCityID=25; mt_xid=V2_52007VwsQU1xaVVoaSClUA2YLEAdbWk5YSk9MQAA0BBZOVQ0ADwNLGlUAZwQXVQpaAlkvShhcDHsCFU5eXENaGkIZWg5nAyJQbVhiWR9BGlUNZwoWYl1dVF0%3D; __jdc=122270672; shshshfp=72ec41b59960ea9a26956307465948f6; rkv=V0700; __jda=122270672.930036140.-.1529979524.1529984840.85; __jdb=122270672.1.930036140|85.1529984840; shshshsID=f797fbad20f4e576e9c30d1c381ecbb1_1_1529984840145'
            }
    r = requests.get(url,headers=head,timeout=600)

    #指定编码方式，不然会出现乱码
    r.encoding='utf-8'
    html1 = etree.HTML(r.text)
    #print (r.text)
    #定位到每一个商品标签li
    datas=html1.xpath('//li[contains(@class,"gl-item")]')
    #print (r.text)
    currentPages=[]
    #print (lastPages)
    titles=[]
    sameNum=0
    print (len(datas))
    for data in datas:
        p_name=data.xpath('div/div[@class="p-name"]/a/em')
        #p_name=data.xpath('div/div[@class="p-name p-name-type-2"]/a/em')
        title=[p_name[0].xpath('string(.)')][0].strip()
        title=title.replace('\n','')
        line=keyword+'\t'+title
        if title in lastPages:
            sameNum+=1
        titles.append(line)
        currentPages.append(title)
        current_num+=1
    if sameNum>=len(titles)*0.5:
        return False,currentPages,current_num


    fout.write('\n'.join(titles))
    return True,currentPages,current_num


#定义函数抓取每页后30条商品信息
def crow_last(fout,n,keyword,href,lastPages,current_num):
    #获取当前的Unix时间戳，并且保留小数点后5位
    a=time.time()
    b='%.5f'%a
    #url=href+'&enc=utf-8&qrst=1&rt=1&stop=1&page='+str(2*n)+'&s='+str(48*n-20)+'&scrolling=y&log_id='+str(b)
    url='https://search.jd.com/s_new.php?keyword='+keyword.split('--')[2]+'&enc=utf-8&qrst=1&rt=1&stop=1&page='+str(2*n)+'&s='+str(48*n-20)+'&scrolling=y&log_id='+str(b)
    head={'authority': 'search.jd.com',
    'method': 'GET',
    'path': '/s_new.php?keyword=%E6%89%8B%E6%9C%BA&enc=utf-8&qrst=1&rt=1&stop=1&vt=2&wq=%E6%89%8B%E6%9C%BA',
    'scheme':'https',
    'referer': 'https://search.jd.com/Search?keyword=%E6%89%8B%E6%9C%BA&enc=utf-8&qrst=1&rt=1&stop=1&vt=2&wq=%E6%89%8B%E6%9C%BA&cid2=653&cid3=655&page=3&s=58&click=0',
    'user-agent': 'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/66.0.3359.139 Safari/537.36',
    'x-requested-with': 'XMLHttpRequest',
    'Cookie':'qrsc=3; pinId=RAGa4xMoVrs; xtest=1210.cf6b6759; ipLocation=%u5E7F%u4E1C; _jrda=5; TrackID=1aUdbc9HHS2MdEzabuYEyED1iDJaLWwBAfGBfyIHJZCLWKfWaB_KHKIMX9Vj9_2wUakxuSLAO9AFtB2U0SsAD-mXIh5rIfuDiSHSNhZcsJvg; shshshfpa=17943c91-d534-104f-a035-6e1719740bb6-1525571955; shshshfpb=2f200f7c5265e4af999b95b20d90e6618559f7251020a80ea1aee61500; cn=0; 3AB9D23F7A4B3C9B=QFOFIDQSIC7TZDQ7U4RPNYNFQN7S26SFCQQGTC3YU5UZQJZUBNPEXMX7O3R7SIRBTTJ72AXC4S3IJ46ESBLTNHD37U; ipLoc-djd=19-1607-3638-3638.608841570; __jdu=930036140; user-key=31a7628c-a9b2-44b0-8147-f10a9e597d6f; areaId=19; __jdv=122270672|direct|-|none|-|1529893590075; PCSYCityID=25; mt_xid=V2_52007VwsQU1xaVVoaSClUA2YLEAdbWk5YSk9MQAA0BBZOVQ0ADwNLGlUAZwQXVQpaAlkvShhcDHsCFU5eXENaGkIZWg5nAyJQbVhiWR9BGlUNZwoWYl1dVF0%3D; __jdc=122270672; shshshfp=72ec41b59960ea9a26956307465948f6; rkv=V0700; __jda=122270672.930036140.-.1529979524.1529984840.85; __jdb=122270672.1.930036140|85.1529984840; shshshsID=f797fbad20f4e576e9c30d1c381ecbb1_1_1529984840145'
 
    }
    r = requests.get(url,headers=head,timeout=600)
    #指定编码方式，不然会出现乱码
    r.encoding='utf-8'
    html1 = etree.HTML(r.text)
    #print (html1)
    #print (r.text)
    #定位到每一个商品标签li
    datas=html1.xpath('//li[contains(@class,"gl-item")]')
    #print (r.text)
    currentPages=[]
    #print (lastPages)
    titles=[]
    sameNum=0
    for data in datas:
        # p_comment = data.xpath('div/div[5]/strong/a/text()')
        #p-name p-name-type-2   "p-name p-name-type-2
        #
        p_name=data.xpath('div/div[@class="p-name"]/a/em')
        #p_name=data.xpath('div/div[@class="p-name p-name-type-2"]/a/em')
        title=[p_name[0].xpath('string(.)')][0].strip()
        title=title.replace('\n','')
        line=keyword+'\t'+title
        if title in lastPages:
            sameNum+=1
        titles.append(line)
        currentPages.append(title)
        current_num+=1
    #print (titles)
    #print (sameNum)
    if sameNum>=len(titles)*0.5:
        return False,currentPages,current_num


    fout.write('\n'.join(titles))
    return True,currentPages,current_num

 
def single_thread(readlines,cate_num,cates):
    current_cate_num={}
    num=0
    for i in range(0,len(readlines)):
        word=readlines[i].strip().split('----')[0].strip()
        num+=1
        href=readlines[i].strip().split('----')[1].strip()
        num=cate_num[word]
        print ('正在爬取第: '+str(i)+'个: '+word)
        print ('需要爬取: '+str(num))
        wrong_num=0
        lastPages=[]
        j=0
        repeatNum=0
        current_num=0
        #continue
        while True:
            j+=1
            #下面的print函数主要是为了方便查看当前抓到第几页了
            if j%10==0:
                print (current_num)
            try:
                #print('   First_Page:   ' + str(i))
                last_num=current_num
                isright,lastPages,current_num=crow_first(fout,j,word,href,lastPages,current_num)
                #isright,lastPages,current_num=crow_last(fout,j,word,href,lastPages,current_num)#通过链接爬虫不需要爬去下半页
                if current_num-last_num<=5:
                    repeatNum+=1
                #crow_last(fout,j,word,href)
                #print('   Finish')
            except Exception as e:
                print (e)
                wrong_num+=1
            '''
            单个分类结束的条件，需要改进
            '''
            if current_num>=cate_num[word] or repeatNum>=5 or wrong_num>=5:
                current_cate_num[word]=current_num
                break
        '''
        记录爬去过程中错误信息
        '''
        if current_cate_num[word]<=0.8*cate_num[word]:
            ferror.write(word+'\t'+str(cate_num[word])+'\t'+str(current_cate_num[word])+'\n')
        fout.close()
        ferror.close()    

    print ('单线程完成')      


if __name__=='__main__':
    '''

    '''
    fin=open('data/train-ubuntu.tsv')
    cate_num={}
    current_num={}
    readlines=fin.readlines()[1:]
    for i in range(len(readlines)):
        line=readlines[i].strip()
        label=line.split('\t')[1]
        if label not in cate_num.keys():
            cate_num[label]=1
        else:
            cate_num[label]+=1
    fin.close()



    fin=open('1258类分类链接.txt','r')#存放1253个类别对应的链接
    readlines=fin.readlines()
    #目前只能使用一个线程
    num_thread=1
    perlines=int((len(readlines)-1)/num_thread+1)
    start=0
    end=start+perlines
    single_thread(readlines,cate_num,cates)


