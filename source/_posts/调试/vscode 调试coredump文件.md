---
title: vscode 调试coredump 文件
index_img: https://s2.loli.net/2024/07/30/AlomdV8WieyhIcQ.png
tags: [frp,内网穿透]
date: 2024-07-30 18:42:30
categories: 工具配置
published: true
comment: 'utterances'
---

1. 设置`ulimit -c unlimited`，如果`ulimit -c `结果是0的话产生不了coredump文件

    ```shell
          ulimit -c [size]  //这里size一般修改为unlimited,或者是其他数字：2048
    ```


    ​上修改只对当前的shell有效，一旦关闭，则恢复原来的值


2. `cat /var/log/apport.log` 可以看到生成的日志信息

3. core文件路径

   - ubuntu20的生成coredump路径不在可执行路径下，而是在`/var/lib/apport/coredump`，因为没有写入权限，所以产生不了coredump文件，需要sudo，或者修改产生路径。

   - 修改core文件产生位置在可执行文件目录下

     `sudo bash -c 'echo core.%e.%p > /proc/sys/kernel/core_pattern'`



4. vscode 配置

   - gdb

     ```json
             {
                 "name": "(gdb) Launch",
                 "type": "cppdbg",
                 "request": "launch",
                 "program": "${workspaceFolder}/a.out",
                 "args": [],
                 "stopAtEntry": false,
                 "cwd": "${workspaceFolder}",
                 "environment": [],
                 "externalConsole": false,
                 "MIMode": "gdb",
                 "setupCommands": [
                     {
                         "description": "Enable pretty-printing for gdb",
                         "text": "-enable-pretty-printing",
                         "ignoreFailures": true
                     }
                 ],
                 "coreDumpPath": "${workspaceFolder}/core.a.out.372125"
             },
     ```


    虽然填了program，但是实际上是从coredump启动的

   - codelldb

     ```json
      {
                 "type": "lldb",
                 "request": "custom",
                 "name": "Open a core dump",
                 "initCommands": [
                     "target create -c ${workspaceFolder}/core.a.out.372125"
                 ]
      }
     ```

     

​			