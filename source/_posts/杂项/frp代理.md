---
title: frp 内网穿透配置
index_img: https://s2.loli.net/2024/07/30/AlomdV8WieyhIcQ.png
tags: [frp,内网穿透]
date: 2024-07-30 18:42:30
categories: 工具配置
published: true
comment: 'utterances'
---

# frp 内网穿透配置

> 要放暑假了，在家需要连接实验室服务器，之前通过云主机搭建frp中转，但是云主机比较贵，所以来白嫖Sakura Frp

## 1. 注册账户

普通用户有两个隧道，限速10M,一个月有10G流量，可以每日签到获取额外的流量，不过对于ssh来讲，基本上是够用的，代码文本也消耗不了多少流量。

## 2. 建立隧道

实名制后，可以点击服务来创建一个隧道，来中转服务。

![QQ_1722343688110](https://s2.loli.net/2024/07/30/uOVJQl7mUSLnt2C.png)

这样就创建了一个隧道

frp本质上做的事情就是，将两端的流量进行转发，因为两端没有公网IP，也可以使用`zerotier`

来打洞，但是如果NAT层比较多，延时还是比较大，在学校里面使用延时比较低，如果在家延时比较高，体验不好。



## 3. 服务端

我把你要访问的电脑称为服务端，也就是实验室的服务器

服务端需要下载frp

![QQ_1722342725796](https://s2.loli.net/2024/07/30/LdeWwjQl34Tn9r7.png)

- 点击复制链接

```shell
wget 链接
```

- 将frp下载下来，使用`tar -xvf natfrp-service_linux_amd64.tar.zst `解压缩

- `chmod +x ./frpc`来赋予可执行权限

- 获取加密参数

  ![QQ_1722342962407](https://s2.loli.net/2024/07/30/7W1t2ougqnmKOTQ.png)



- 启动

  ` ./frpc -f 参数`

  - 可以使用`systemctl`来管理frp的服务，包括开机自动启动，开始，暂停等
  - nohup开控制frpc后端执行 `nohup ./frpc -f 参数 > runoob.log 2>&1 &`

  ![QQ_1722343190952](https://s2.loli.net/2024/07/30/IgwdNfDqC9jY7sR.png)

可以获得访问服务器的`ip:port`

## 4. 客户端

- 下载认证

  ![QQ_1722343257332](https://s2.loli.net/2024/07/30/EhS36gJaFVGfLle.png)

![QQ_1722343281279](https://s2.loli.net/2024/07/30/LRFC9bnuv1jlDZc.png)

会生成一个exe，点击运行即可

- ssh 连接

  注意这里ssh端口不是22

  而是上面获得的那个IP和端口

  ```shell
  ssh -p port yourname@ip
  ```

  这样就连接到服务器了。

